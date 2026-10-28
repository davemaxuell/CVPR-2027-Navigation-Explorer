"""Portable recordings preserve evidence and reject ambiguous/corrupt input."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from navigation_explorer.investigations import build_case, build_queue
from navigation_explorer.recordings import EPISODE_SCHEMA, PortableRun, normalize_episode, read_records
from navigation_explorer.writer import episode_from_positions, write_recordings


def basic(**changes):
    value = {'schema_version': EPISODE_SCHEMA, 'instr_id': 'episode-1', 'scene': 'room',
             'instruction': 'Visit the door.',
             'trajectory': [{'node_id': 'a', 'position': [0, 0, 0]}, {'node_id': 'b', 'position': [2, 0, 0]}]}
    value.update(changes)
    return value


def normalize(record, **kwargs):
    return normalize_episode(record, run_name='model/validation', environment='continuous', **kwargs)


def pointer(bundle, path):
    value = bundle
    for part in path.strip('/').split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class PortableRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_file(self, records, **changes):
        path = self.directory / 'episodes.jsonl'
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        spec = {'id': 'model/validation', 'adapter': 'recording', 'model': 'Other model',
                'dataset': 'Other dataset', 'split': 'validation', 'environment': 'continuous',
                'episodes': path.name, **changes}
        return PortableRun(spec, self.directory, self.directory / 'state')

    def test_bundled_demo_covers_discrete_continuous_unscored(self):
        demo = Path(__file__).parents[1] / 'src/navigation_explorer/demo'
        manifest = json.loads((demo / 'project.json').read_text())
        environments, outcomes, signals = set(), set(), set()
        for spec in manifest['runs']:
            run = PortableRun(spec, demo, self.directory)
            environments.add(run.environment)
            info = {'name': run.name, 'env': run.environment}
            queue = build_queue(info, run.rows())
            for case in queue['cases']:
                signals.update(flag['kind'] for flag in case['flags'])
            for instr_id in run.ids:
                ep = run.episode(instr_id)
                outcomes.add(ep['outcome'])
                case = build_case(info, ep)
                json.dumps(case, allow_nan=False)
                self.assertEqual(case['images'], [])
                for finding in case['findings']:
                    frame = finding['frame']
                    self.assertTrue(frame is None or 0 <= frame < len(ep['frames']))
                    for evidence in finding['evidence']:
                        self.assertEqual(pointer(case, evidence['path']), evidence['value'])
        self.assertEqual(environments, {'discrete', 'continuous'})
        self.assertTrue({'success', 'wrong_stop', 'step_limit', 'unscored'} <= outcomes)
        self.assertTrue({'missed_stop', 'inefficient_success', 'blocked_movement', 'waypoint_miss', 'evidence_gap'} <= signals)

    def test_unscored_trace_has_no_inferred_failure_stop_or_decision_count(self):
        ep = normalize(basic())
        self.assertEqual(ep['outcome'], 'unscored')
        self.assertNotIn('sr', ep['metrics'])
        self.assertIsNone(ep['metrics']['decisions'])
        self.assertIsNone(ep['metrics']['moves'])
        self.assertEqual(ep['decisions'][0]['action'], 'unknown')
        self.assertEqual(ep['decisions'][0]['source'], 'replay_placeholder')
        self.assertFalse(ep['capabilities']['decisions'])
        self.assertEqual(ep['termination'], 'unknown')
        self.assertFalse(ep['diag']['steplog'])

    def test_one_frame_without_actions_is_a_valid_recording(self):
        ep = normalize(basic(trajectory=[{'position': [0, 0, 0]}]))
        self.assertEqual(ep['metrics']['tl'], 0)
        self.assertEqual(len(ep['frames']), 1)
        self.assertIsNone(ep['metrics']['moves'])

    def test_success_is_not_inferred_from_position_or_distance(self):
        record = basic(goal={'position': [2, 0, 0]}, evaluation={'distance_kind': 'geodesic', 'success_radius_m': 3},
                       metrics={'ne': 0})
        record['trajectory'][0]['distance_to_goal'] = 2
        record['trajectory'][1]['distance_to_goal'] = 0
        ep = normalize(record)
        self.assertEqual(ep['outcome'], 'unscored')
        self.assertNotIn('sr', ep['metrics'])
        self.assertNotIn('osr', ep['metrics'])
        self.assertEqual(ep['diag']['near_goal_hops'], 2)

    def test_euclidean_geometry_is_not_substituted_for_missing_geodesics(self):
        ep = normalize(basic(goal={'position': [2, 0, 0]}, evaluation={'distance_kind': 'geodesic', 'success_radius_m': 1}))
        self.assertIsNone(ep['frames'][0]['dist'])
        self.assertEqual(ep['frames'][0]['euclid'], 2)
        self.assertNotIn('closest_hop', ep['diag'])
        self.assertNotIn('near_goal_hops', ep['diag'])
        self.assertEqual(ep['goal_zone'], [])

    def test_recorded_metrics_and_derived_sample_length_have_distinct_sources(self):
        ep = normalize(basic(metrics={'sr': 1, 'spl': .3, 'tl': 8, 'custom_score': 42}))
        case = build_case({'name': 'a'}, ep)
        self.assertEqual(ep['metrics']['tl'], 8)
        for key in ('sr', 'spl', 'tl', 'custom_score'):
            self.assertEqual(case['provenance']['metric_sources'][key], 'recorded_metric')
        self.assertEqual(case['provenance']['metric_sources']['hops'], 'trace_derived_metric')
        sampled = normalize(basic())
        self.assertEqual(sampled['metric_sources']['tl'], 'sampled_trace_derived_metric')
        self.assertIn('underestimate', sampled['metric_notes']['tl'])

    def test_partial_goal_distance_coverage_does_not_claim_unsampled_visit(self):
        record = basic(evaluation={'distance_kind': 'geodesic', 'success_radius_m': 1}, metrics={'sr': 0, 'osr': 1})
        record['trajectory'][0]['distance_to_goal'] = 2
        ep = normalize(record)
        self.assertFalse(ep['diag']['distance_coverage_complete'])
        self.assertNotIn('goal_visit_unsampled', ep['diag'])
        self.assertNotIn('near_goal_hops', ep['diag'])
        case = build_case({'name': 'a'}, ep)
        finding = next(f for f in case['findings'] if f['kind'] == 'missed_stop')
        self.assertIsNone(finding['frame'])
        self.assertIsNone(finding['decision'])

    def test_dense_oracle_success_does_not_invent_frame(self):
        record = basic(evaluation={'distance_kind': 'geodesic', 'success_radius_m': 1}, metrics={'sr': 0, 'osr': 1})
        for point in record['trajectory']:
            point['distance_to_goal'] = 3
        ep = normalize(record)
        self.assertTrue(ep['diag']['goal_visit_unsampled'])
        case = build_case({'name': 'a'}, ep)
        finding = next(f for f in case['findings'] if f['kind'] == 'missed_stop')
        self.assertIsNone(finding['frame'])
        self.assertIsNone(finding['decision'])

    def test_failed_metric_without_termination_has_neutral_failure_category(self):
        self.assertEqual(normalize(basic(metrics={'sr': 0}))['outcome'], 'failure')
        self.assertEqual(normalize(basic(metrics={'sr': 0}, termination='stop'))['outcome'], 'wrong_stop')
        self.assertEqual(normalize(basic(metrics={'sr': 0}, termination='step_limit'))['outcome'], 'step_limit')
        self.assertEqual(normalize(basic(termination='step_limit'))['outcome'], 'unscored')

    def test_raw_logits_are_not_converted_to_probabilities(self):
        record = basic(decisions=[{'frame': 0, 'action': 'move', 'target': 'b',
                                  'options': [{'node_id': 'b', 'score': 15}, {'node_id': None, 'score': -5}]}])
        ep = normalize(record)
        self.assertNotIn('model', ep['decisions'][0])
        self.assertEqual(ep['decisions'][0]['logged_options'][0]['score'], 15)
        self.assertIsNone(ep['decisions'][0]['stop_prob'])
        self.assertEqual(ep['reference'], 'none')

    def test_partial_probability_distribution_does_not_invent_stop_probability(self):
        ep = normalize(basic(decisions=[{'frame': 0, 'action': 'move', 'target': 'b',
                                        'options': [{'node_id': 'b', 'probability': .9}]}]))
        self.assertIsNone(ep['decisions'][0]['model']['fused'][0])
        self.assertNotIn('ref', ep['decisions'][0])

    def test_net_displacement_and_waypoint_miss_use_recorded_positions(self):
        record = basic(trajectory=[{'position': [0, 0, 0]}, {'position': [.1, 0, 0]}],
                       decisions=[{'frame': 0, 'action': 'move', 'target': {'position': [2, 0, 0]}}])
        ep = normalize(record)
        self.assertEqual(ep['diag']['stuck'], 1)
        self.assertAlmostEqual(ep['diag']['ctrl_miss_max'], 1.9)
        case = build_case({'name': 'a'}, ep)
        self.assertEqual({f['kind'] for f in case['findings']}, {'blocked_movement', 'waypoint_miss'})

    def test_discrete_recording_without_images_does_not_fabricate_skyboxes(self):
        ep = normalize_episode(basic(), run_name='x', environment='discrete')
        case = build_case({'name': 'a'}, ep)
        self.assertEqual(case['images'], [])
        self.assertEqual(case['provenance']['images']['kind'], 'unavailable')

    def test_optional_images_keep_declared_provenance_and_contain_paths(self):
        images = self.directory / 'images'
        (images / 'episode').mkdir(parents=True)
        image = images / 'episode/frame_0.jpg'
        image.write_bytes(b'synthetic test file; not decoded')
        record = basic(image_provenance={'kind': 'regenerated_preview', 'label': 'Declared preview'})
        record['trajectory'][0]['image_prefix'] = 'episode/frame'
        run = self.run_file([record], images_dir='images')
        ep = run.episode('episode-1')
        case = build_case({'name': run.name}, ep)
        self.assertEqual(len(case['images']), 1)
        self.assertEqual(len(case['images'][0]['urls']), 6)
        self.assertIn('model%2Fvalidation/episode%2Fframe_0.jpg', case['images'][0]['urls'][0])
        self.assertFalse(case['provenance']['images']['independently_verified'])
        # Asset paths are canonical, including Windows short-name temp directories.
        self.assertEqual(run.image_file('episode/frame_0.jpg'), image.resolve())
        self.assertIsNone(run.image_file('../episodes.jsonl'))
        self.assertIsNone(run.image_file('/etc/passwd'))
        self.assertIsNone(run.image_file('frame.png'))
        outside = self.directory / 'outside.jpg'
        outside.write_bytes(b'outside')
        try:
            (images / 'escape.jpg').symlink_to(outside)
        except (OSError, NotImplementedError):
            if os.name != 'nt':
                raise
            # Windows runners may not grant symlink privileges. The ordinary
            # containment assertions above still run on every platform.
        else:
            self.assertIsNone(run.image_file('escape.jpg'))

    def test_invalid_shapes_references_and_metadata_are_actionable(self):
        cases = [basic(trajectory=[]), basic(trajectory=[{'position': [0, 1]}]),
                 basic(trajectory=[{'position': [0, 1, float('nan')]}]),
                 basic(trajectory=[{'node_id': 'missing'}]),
                 basic(graph={'edges': [['a', 'missing']]}),
                 basic(trajectory=[{'node_id': 'x', 'position': [0, 0, 0]}, {'node_id': 'x', 'position': [1, 0, 0]}]),
                 basic(metrics={'sr': True}), basic(metrics={'sr': .5}), basic(metrics={'spl': 1.1}),
                 basic(metrics={'stuck': 10}), basic(metrics={'instr_id': 1}),
                 basic(metadata={'nested': {'overflow': float('inf')}}), basic(metadata=[])]
        for record in cases:
            with self.subTest(record=record), self.assertRaises(ValueError):
                normalize(record)

    def test_decision_alignment_and_option_validation(self):
        cases = [[{'frame': 1, 'action': 'stop'}],
                 [{'frame': 0, 'action': 'move'}, {'frame': 0, 'action': 'stop'}],
                 [{'frame': 0, 'action': 'move', 'target': 'missing'}],
                 [{'frame': 0, 'action': 'stop', 'end_frame': 0}],
                 [{'frame': 0, 'action': 'move', 'options': [{'probability': .5}]}],
                 [{'frame': 0, 'action': 'move', 'options': [{'node_id': None, 'probability': .9}, {'node_id': 'b', 'probability': .9}]}],
                 [{'frame': 0, 'action': 'move', 'options': [{'node_id': 'b', 'probability': .2}, {'node_id': 'b', 'probability': .3}]}]]
        for decisions in cases:
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                normalize(basic(decisions=decisions))

    def test_malformed_json_duplicate_ids_and_overflow_fail(self):
        with self.assertRaisesRegex(ValueError, 'duplicate instr_id'):
            self.run_file([basic(), basic()])
        path = self.directory / 'invalid.jsonl'
        path.write_text('{"instr_id":"a","instr_id":"b"}\n')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            read_records(path)
        path.write_text('{"value":NaN}\n')
        with self.assertRaises(ValueError):
            read_records(path)
        record = basic(metadata={'overflow': 1})
        path.write_text(json.dumps(record).replace('"overflow": 1', '"overflow": 1e999'))
        with self.assertRaises(ValueError):
            normalize(read_records(path)[0])

    def test_invalid_timestamps_floors_and_image_prefix(self):
        cases = [basic(trajectory=[{'position': [0, 0, 0], 'floor': 3}]),
                 basic(trajectory=[{'position': [0, 0, 0], 'time_s': 2}, {'position': [1, 0, 0], 'time_s': 1}]),
                 basic(trajectory=[{'position': [0, 0, 0], 'image_prefix': '../escape'}])]
        for record in cases:
            with self.subTest(record=record), self.assertRaises(ValueError):
                normalize(record)

    def test_episodes_and_rows_are_isolated_copies(self):
        run = self.run_file([basic(metrics={'sr': 0})])
        ep = run.episode('episode-1')
        ep['metrics']['sr'] = 1
        ep['frames'][0]['vp'] = 'changed'
        run.rows()[0]['sr'] = 1
        self.assertEqual(run.episode('episode-1')['metrics']['sr'], 0)
        self.assertEqual(run.episode('episode-1')['frames'][0]['vp'], 'a')
        self.assertEqual(run.rows()[0]['sr'], 0)

    def test_writer_validates_before_publication_and_requires_explicit_overwrite(self):
        record = episode_from_positions('x', 'scene', 'Walk.', [(0, 0, 0), (1, 0, 0)])
        path = self.directory / 'recordings.jsonl'
        write_recordings(path, [record], environment='continuous')
        previous = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_recordings(path, [basic()], environment='continuous')
        self.assertEqual(path.read_bytes(), previous)
        with self.assertRaises(ValueError):
            write_recordings(path, [basic(trajectory=[])], environment='continuous', overwrite=True)
        self.assertEqual(path.read_bytes(), previous)
        write_recordings(path, [basic()], environment='continuous', overwrite=True)
        self.assertEqual(read_records(path)[0]['instr_id'], 'episode-1')

    def test_run_metadata_type_is_validated(self):
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.run_file([basic()], metadata=[])
        with self.assertRaisesRegex(ValueError, 'images_dir'):
            self.run_file([basic()], images_dir=4)

    def test_explicit_matterport_coordinates_can_pair_with_native_runs(self):
        record = basic(coordinate_system='matterport_z_up_m')
        run = self.run_file([record], coordinate_system='matterport_z_up_m', comparison_group='task-aligned')
        self.assertEqual(run.coordinate_system, 'matterport_z_up_m')
        self.assertEqual(run.comparison_group, 'task-aligned')
        self.assertEqual(run.episode('episode-1')['coordinate_system'], 'matterport_z_up_m')
        self.assertEqual(run.episode('episode-1')['graph']['nodes']['b']['pos'], [2, 0, 0])
        inherited = self.run_file([basic()], coordinate_system='matterport_z_up_m')
        self.assertEqual(inherited.episode('episode-1')['coordinate_system'], 'matterport_z_up_m')
        with self.assertRaisesRegex(ValueError, 'must match'):
            self.run_file([record])
        with self.assertRaisesRegex(ValueError, 'coordinate_system'):
            self.run_file([basic()], coordinate_system='habitat_y_up')


if __name__ == '__main__':
    unittest.main()

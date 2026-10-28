"""Native-format portability checks with self-contained synthetic recordings."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from navigation_explorer import parse_trajectories as pt
from navigation_explorer.legacy import load_legacy_run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def connectivity():
    result = []
    for index, name in enumerate(('a', 'b', 'c')):
        pose = [1, 0, 0, index * 4, 0, 1, 0, 0, 0, 0, 1, 1.5, 0, 0, 0, 1]
        result.append({'image_id': name, 'included': True, 'pose': pose, 'height': 1.5,
                       'unobstructed': [abs(index - other) == 1 for other in range(3)]})
    return result


def ce_record():
    return {
        'episode_id': 7, 'instr_id': 'custom-episode', 'trajectory_id': 17,
        'instruction': 'Walk to the end.', 'scene': 'room', 'camera_height': 1.5,
        'nodes': {'n0': [0, 0, 0], 'n1': [4, 0, 0]}, 'ghosts': {'g0': [4, 0, 0]},
        'ghost_fronts': {'g0': ['n0']}, 'track': [[0, 0, 0], [4, 0, 0]],
        'samples': [0, 1], 'seg_end': [1, 2, 2], 'track_goal_dist': [8, 4],
        'reference_path': [[0, 0, 0], [8, 0, 0]], 'gt_locations': [[0, 0, 0], [8, 0, 0]],
        'ref_geo': [[0, 8], [8, 0]], 'geo': {
            'node': {'n0': [0, 8], 'n1': [4, 4]}, 'ghost': {'g0': [4, 4]}, 'sample': [[0, 8], [4, 4]]},
        'max_steps': 2, 'tokens': [], 'views': True,
        'steps': [
            {'vp': 'n0', 'front': 'n0', 'choice': 'g0', 'heading': 0,
             'gmap': ['n0', 'g0'], 'visited': [True, False], 'fused': [.1, 0, .9], 'cands': []},
            {'vp': 'n1', 'front': 'n1', 'choice': None, 'heading': 0, 'stop_at': 'n1',
             'gmap': ['n0', 'n1', 'g0'], 'visited': [True, True, False], 'fused': [.8, 0, 0, .2],
             'stop_scores': {'n1': .8}, 'cands': []}],
        'metrics': {'distance_to_goal': 4, 'success': 0, 'oracle_success': 0, 'spl': 0,
                    'ndtw': .5, 'sdtw': 0, 'path_length': 4, 'collisions': 0, 'steps_taken': 2}}


class LegacyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'project'
        self.state = Path(self.tmp.name) / 'state'
        write_json(self.root / 'connectivity/room_connectivity.json', connectivity())
        write_json(self.root / 'annotations.json', [{'path_id': 17, 'scan': 'room', 'path': ['a', 'b', 'c'],
                                                    'instructions': ['Walk to the end.']}])
        write_json(self.root / 'predictions.json', [{'instr_id': '17_0', 'trajectory': [['a'], ['b']]}])
        self.r2r = {'id': 'custom/model', 'adapter': 'r2r', 'model': 'A new model', 'dataset': 'My R2R',
                    'split': 'new-evaluation', 'predictions': 'predictions.json',
                    'annotations': 'annotations.json', 'connectivity': 'connectivity'}

    def ce_spec(self, record=None):
        # Ordinary json.dumps includes spaces, and episode_id is an integer.
        (self.root / 'ce.jsonl').write_text(json.dumps(record or ce_record()) + '\n')
        return {'id': 'CE model / custom split', 'adapter': 'etpnav', 'model': 'Custom ETPNav',
                'dataset': 'R2R-CE', 'split': 'new evaluation', 'episodes': 'ce.jsonl',
                'connectivity': 'connectivity'}

    def test_r2r_needs_no_checkout_and_embeds_graph(self):
        run = load_legacy_run(self.r2r, self.root, self.state)
        episode = run.episode('17_0')
        self.assertEqual(run.dataset, 'My R2R')
        self.assertEqual(episode['metrics']['ne'], 4)
        self.assertEqual(episode['metrics']['sr'], 0)
        self.assertEqual(episode['graph']['nodes']['b']['pos'], [4, 0, 1.5])
        self.assertEqual(episode['goal_zone'], ['c'])
        self.assertEqual(episode['image_provenance']['kind'], 'unknown')
        self.assertTrue(all(not n['rot'] for n in episode['graph']['nodes'].values()))
        self.assertEqual(run.coordinate_system, 'matterport_z_up_m')
        self.assertFalse(self.state.exists())  # Reading an episode does not write caches.
        run.rows()
        self.assertTrue(list(self.state.rglob('*.json')))

    def test_r2r_cache_invalidates_when_annotations_or_connectivity_change(self):
        run = load_legacy_run(self.r2r, self.root, self.state)
        self.assertEqual(run.rows()[0]['ne'], 4)
        changed = connectivity()
        changed[2]['pose'][3] = 12
        write_json(self.root / 'connectivity/room_connectivity.json', changed)
        self.assertEqual(load_legacy_run(self.r2r, self.root, self.state).rows()[0]['ne'], 8)
        write_json(self.root / 'annotations.json', [{'path_id': 17, 'scan': 'room', 'path': ['a', 'b'],
                                                    'instructions': ['Walk to the end.']}])
        self.assertEqual(load_legacy_run(self.r2r, self.root, self.state).rows()[0]['sr'], 1)

    def test_missing_inputs_report_manifest_field(self):
        for field in ('predictions', 'annotations', 'connectivity'):
            spec = dict(self.r2r)
            del spec[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                load_legacy_run(spec, self.root, self.state)

    def test_native_coordinates_cannot_be_relabelled(self):
        spec = dict(self.r2r, coordinate_system='habitat_y_up')
        with self.assertRaisesRegex(ValueError, 'coordinate_system'):
            load_legacy_run(spec, self.root, self.state)

    def test_utf8_annotations_metadata_and_csv_export(self):
        instruction = '문으로 이동하세요. 前往门口。'
        write_json(self.root / 'annotations.json', [{'path_id': 17, 'scan': 'room', 'path': ['a', 'b', 'c'],
                                                    'instructions': [instruction]}])
        write_json(self.root / 'metadata.json', {'note': '다국어 실험'})
        run = load_legacy_run(dict(self.r2r, metadata='metadata.json'), self.root, self.state)
        self.assertEqual(run.rows()[0]['instruction'], instruction)
        self.assertEqual(run.note, '다국어 실험')
        exported = pt.write_export(run, run.rows(), self.root / 'review.csv', 'csv')
        self.assertIn(instruction, exported.read_bytes().decode('utf-8'))

    def test_object_navigation_requires_explicit_goal_visibility(self):
        write_json(self.root / 'annotations.json', [{'path_id': 17, 'scan': 'room', 'path': ['a', 'b', 'c'],
                                                    'instructions': ['Find the chair.'], 'objId': '42'}])
        with self.assertRaisesRegex(ValueError, 'bboxes'):
            load_legacy_run(self.r2r, self.root, self.state)
        write_json(self.root / 'BBoxes.json', {'room_c': {'42': {'visible_pos': [0]}}})
        run = load_legacy_run(self.r2r, self.root, self.state)
        self.assertIsNone(run.evaluation['success_radius_m'])
        self.assertEqual(run.evaluation['success_criterion'], 'visible_goal_viewpoints')
        self.assertEqual(run.episode('17_0')['goal_zone'], ['c'])

    def test_ce_only_custom_split_and_standard_json_spacing(self):
        spec = self.ce_spec()
        run = load_legacy_run(spec, self.root, self.state)
        self.assertEqual(run.ids, ['custom-episode'])
        self.assertEqual(run.split, 'new evaluation')
        self.assertTrue(run.ce)
        self.assertEqual(run.max_action_len, 2)
        episode = run.episode('custom-episode')
        self.assertEqual(episode['metrics']['ne'], 4)
        self.assertEqual(episode['metrics']['sr'], 0)
        self.assertEqual(episode['graph']['nodes']['n1']['pos'], [4, 0, 1.5])
        self.assertEqual(episode['ce']['image_provenance']['kind'], 'unknown')
        self.assertFalse(any('sky' in n for n in episode['graph']['nodes'].values()))
        self.assertEqual(run.rows()[0]['outcome'], 'step_limit')

    def test_ce_annotation_mapping_is_optional_and_explicit(self):
        record = ce_record()
        del record['instr_id']
        spec = self.ce_spec(record)
        self.assertEqual(load_legacy_run(spec, self.root, self.state).ids, ['7'])
        spec['annotations'] = 'annotations.json'
        self.assertEqual(load_legacy_run(spec, self.root, self.state).ids, ['17_0'])

    def test_ce_cache_invalidates_on_same_size_record_replacement(self):
        spec = self.ce_spec()
        first = load_legacy_run(spec, self.root, self.state)
        self.assertEqual(first.rows()[0]['ne'], 4)
        record = ce_record()
        record['metrics']['distance_to_goal'] = 5
        self.ce_spec(record)
        self.assertTrue(first.stale())
        self.assertEqual(load_legacy_run(spec, self.root, self.state).rows()[0]['ne'], 5)

    def test_duplicate_ce_identity_and_missing_native_fields_are_rejected(self):
        spec = self.ce_spec()
        text = (self.root / 'ce.jsonl').read_text()
        (self.root / 'ce.jsonl').write_text(text + text)
        with self.assertRaisesRegex(ValueError, 'duplicate instruction ID'):
            load_legacy_run(spec, self.root, self.state)
        record = ce_record()
        del record['geo']
        self.ce_spec(record)
        with self.assertRaisesRegex(ValueError, 'geo'):
            load_legacy_run(spec, self.root, self.state)

    def test_explicit_native_images_have_encoded_run_urls_and_containment(self):
        spec = self.ce_spec()
        (self.root / 'views/7').mkdir(parents=True)
        spec['views'] = 'views'
        run = load_legacy_run(spec, self.root, self.state)
        sky = run.episode('custom-episode')['graph']['nodes']['n0']['sky']
        self.assertEqual(sky, '/api/legacy-view/CE%20model%20%2F%20custom%20split/7/n0')
        # Asset paths are canonical, including Windows short-name temp directories.
        self.assertEqual(run.view_file('7', 'n0', 0), (self.root / 'views/7/n0_0.jpg').resolve())
        with self.assertRaisesRegex(ValueError, 'identifier'):
            run.view_file('../outside', 'n0', 0)
        try:
            (self.root / 'views/7/n0_0.jpg').symlink_to(self.root / 'predictions.json')
        except (OSError, NotImplementedError):
            self.skipTest('Symlink creation unavailable; ordinary containment assertions already passed')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            run.view_file('7', 'n0', 0)

    def test_provenance_needs_matching_recording_hash(self):
        spec = self.ce_spec()
        spec['image_provenance'] = {'kind': 'regenerated_preview', 'input_log_sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_legacy_run(spec, self.root, self.state)
        spec['image_provenance']['input_log_sha256'] = hashlib.sha256((self.root / 'ce.jsonl').read_bytes()).hexdigest()
        run = load_legacy_run(spec, self.root, self.state)
        self.assertEqual(run.image_provenance['kind'], 'regenerated_preview')
        self.assertIn('not independently verified', run.image_provenance['verification_scope'])

    def test_aggregate_preserves_missing_metrics_and_denominators(self):
        result = pt.aggregate([{'sr': 1, 'ne': 2}, {'sr': None, 'spl': .3}, {'sr': 0}, {}])
        self.assertEqual(result['sr'], 50)
        self.assertEqual(result['ne'], 2)
        self.assertEqual(result['spl'], 30)
        self.assertEqual(result['metric_counts'], {'sr': 2, 'spl': 1, 'ne': 1})
        self.assertNotIn('ndtw', result)


if __name__ == '__main__':
    unittest.main()

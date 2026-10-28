"""Evidence integrity checks for investigation rules (standard library only)."""
import copy
import unittest

from navigation_explorer.investigations import CATEGORIES, build_case, build_queue


RUN = {'name': 'test/val_unseen', 'model': 'test', 'split': 'val_unseen', 'env': 'discrete'}


def episode(**changes):
    value = {
        'instr_id': 'example_0', 'scan': 'scan', 'instruction': 'Go to the door. Stop there.',
        'outcome': 'wrong_stop', 'failure_mode': 'passed goal: did not stop there',
        'metrics': {'sr': 0, 'osr': 1, 'ne': 5.054, 'spl': 0, 'tl': 13.52, 'gt_len': 8.465},
        'diag': {'steplog': True, 'near_goal_hops': 1, 'first_nonbest': 0, 'first_nonbest_conf': 0.58948},
        'frames': [{'vp': 'a', 'decision': 0}, {'vp': 'b', 'decision': 0},
                   {'vp': 'goal', 'decision': 1}, {'vp': 'c', 'decision': 2}],
        'path': ['a', 'b', 'goal', 'c'], 'gt_path': ['a', 'goal'], 'goal': 'goal', 'goal_zone': ['goal'],
        'decisions': [
            {'t': 0, 'at': 'a', 'target': 'goal', 'action': 'local', 'model': {'forced': False},
             'ref': {'in_area': False, 'type': 'wrong move', 'conf': 0.58948}},
            {'t': 1, 'at': 'goal', 'target': 'c', 'action': 'local', 'model': {'forced': False},
             'ref': {'in_area': True, 'type': 'missed stop', 'stop': 0.14741, 'conf': 0.82529}},
            {'t': 2, 'at': 'c', 'target': 'c', 'action': 'stop', 'model': {'forced': False},
             'ref': {'in_area': False, 'type': 'stopped early', 'stop': 0.81352}},
        ],
    }
    value.update(changes)
    return value


def row(ep):
    return dict(ep['metrics'], **ep['diag'], instr_id=ep['instr_id'], scan=ep['scan'],
                outcome=ep['outcome'], instruction=ep['instruction'])


def pointer(data, path):
    current = data
    for part in path.lstrip('/').split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


class InvestigationTests(unittest.TestCase):
    def assertEvidenceResolves(self, case):
        for finding in case['findings']:
            for item in finding['evidence']:
                self.assertEqual(pointer(case, item['path']), item['value'], item)

    def test_goal_visit_anchors_to_missed_stop_decision_not_first_disagreement(self):
        case = build_case(RUN, episode())
        finding = next(f for f in case['findings'] if f['kind'] == 'missed_stop')
        self.assertEqual((finding['decision'], finding['frame']), (1, 2))
        self.assertEqual(finding['hypothesis_status'], 'untested')
        self.assertIn('f=2', finding['replay_url'])
        self.assertIn(0.14741, [e['value'] for e in finding['evidence']])
        self.assertEvidenceResolves(case)

    def test_dense_goal_visit_does_not_invent_replay_frame(self):
        ep = episode(ce={'track': [[0, 0, 0], [1, 0, 0]]}, goal_zone=[])
        ep['diag']['near_goal_hops'] = 0
        ep['diag']['goal_visit_unsampled'] = True
        ep['diag']['first_goal_track_index'] = 1
        ep['diag']['dense_closest_dist'] = 2.9
        for d in ep['decisions']:
            d['ref']['in_area'] = False
            d['ref']['type'] = None
        case = build_case(dict(RUN, env='continuous'), ep)
        goal = next(f for f in case['findings'] if f['kind'] == 'missed_stop')
        gap = next(f for f in case['findings'] if f['kind'] == 'evidence_gap')
        self.assertIsNone(goal['frame'])
        self.assertIsNone(goal['decision'])
        self.assertNotIn('f=', goal['replay_url'])
        self.assertTrue(gap['missing_evidence'])
        self.assertIn('/episode/diagnostics/first_goal_track_index', [e['path'] for e in gap['evidence']])
        self.assertEvidenceResolves(case)

    def test_blocked_movement_uses_execution_and_excludes_stop(self):
        ep = episode(ce={'camera_height': 1.25})
        ep['diag'].update(stuck=1, ctrl_miss_max=1.5)
        ep['decisions'][0]['exec'] = {'moved_m': 0.9, 'miss_m': 0.2}
        ep['decisions'][1]['exec'] = {'moved_m': 0, 'miss_m': 1.5}
        ep['decisions'][2]['exec'] = {'moved_m': 0, 'miss_m': 9}
        case = build_case(RUN, ep)
        blocked = next(f for f in case['findings'] if f['kind'] == 'blocked_movement')
        miss = next(f for f in case['findings'] if f['kind'] == 'waypoint_miss')
        self.assertEqual((blocked['decision'], miss['decision']), (1, 1))
        self.assertEvidenceResolves(case)

    def test_successes_with_detours_remain_in_queue(self):
        ep = episode(outcome='success')
        ep['metrics'].update(sr=1, spl=0.28)
        ep['diag'].update(oscillations=3, first_nonbest_conf=0.9448)
        queue = build_queue(RUN, [row(ep)])
        flags = {f['kind'] for f in queue['cases'][0]['flags']}
        self.assertEqual(flags, {'oscillation', 'inefficient_success', 'confident_reference_disagreement'})
        self.assertEqual(queue['flagged_episodes'], 1)
        self.assertIn('not a probability', queue['priority_note'])
        case = build_case(RUN, ep)
        self.assertNotIn('missed_stop', [f['kind'] for f in case['findings']])
        self.assertEvidenceResolves(case)

    def test_ce_metric_provenance_distinguishes_recorded_spl_context_from_graph_geometry(self):
        ep = episode(outcome='success', ce={'camera_height': 1.25})
        ep['metrics'].update(sr=1, spl=0.28, sp_len=6.2, gt_len=8.465, gt_hops=2,
                             hops=7, moves=4, decisions=5, cls=0.7, euclid_ne=1.2, custom_metric=0.5)
        case = build_case(dict(RUN, env='continuous'), ep)
        finding = next(f for f in case['findings'] if f['kind'] == 'inefficient_success')
        items = {item['path']: item for item in finding['evidence']}
        self.assertEqual(items['/episode/metrics/sp_len']['source'], 'recorded_metric')
        self.assertEqual(items['/episode/metrics/gt_len']['source'], 'reference_derived_metric')
        self.assertIn('CE SPL', items['/episode/metrics/sp_len']['label'])
        for key in ('sr', 'osr', 'spl', 'ne', 'tl', 'sp_len'):
            self.assertEqual(case['provenance']['metric_sources'][key], 'recorded_metric')
        for key in ('gt_len', 'gt_hops', 'cls', 'euclid_ne'):
            self.assertEqual(case['provenance']['metric_sources'][key], 'reference_derived_metric')
        for key in ('hops', 'moves', 'decisions'):
            self.assertEqual(case['provenance']['metric_sources'][key], 'trace_derived_metric')
        self.assertEqual(case['provenance']['metric_sources']['custom_metric'], 'unspecified_metric')
        self.assertEvidenceResolves(case)

    def test_low_net_displacement_keeps_round_trip_explanation_and_distance_walked(self):
        ep = episode(ce={'camera_height': 1.25})
        ep['diag']['stuck'] = 1
        ep['decisions'][1]['exec'] = {'moved_m': 0.1, 'walked_m': 5.0, 'miss_m': 0.2}
        case = build_case(dict(RUN, env='continuous'), ep)
        finding = next(f for f in case['findings'] if f['kind'] == 'blocked_movement')
        self.assertIn('round trip', ' '.join(finding['hypotheses']))
        self.assertIn('does not establish obstruction', ' '.join(finding['missing_evidence']))
        self.assertIn('/trace/decisions/1/exec/walked_m', [item['path'] for item in finding['evidence']])
        self.assertEvidenceResolves(case)

    def test_budget_forced_stop_is_not_reported_as_model_preference(self):
        ep = episode()
        ep['diag'].update(first_nonbest=2, first_nonbest_conf=0.99)
        ep['decisions'][2]['model']['forced'] = True
        case = build_case(RUN, ep)
        self.assertNotIn('confident_reference_disagreement', [f['kind'] for f in case['findings']])
        self.assertEvidenceResolves(case)

    def test_no_ground_truth_or_aligned_log_does_not_invent_scores(self):
        ep = episode(metrics={'tl': 2}, diag={'steplog': False}, goal=None, goal_zone=[], gt_path=[])
        ep['outcome'] = 'no ground truth'
        for d in ep['decisions']:
            d.pop('model')
            d.pop('ref')
        case = build_case(RUN, ep)
        self.assertEqual([f['kind'] for f in case['findings']], ['evidence_gap'])
        self.assertIn('Ground-truth success is unavailable', ' '.join(case['limitations']))
        self.assertEvidenceResolves(case)

    def test_missing_frames_leave_decision_anchor_without_frame(self):
        case = build_case(RUN, episode(frames=[]))
        finding = next(f for f in case['findings'] if f['kind'] == 'missed_stop')
        self.assertEqual(finding['decision'], 1)
        self.assertIsNone(finding['frame'])
        self.assertEvidenceResolves(case)

    def test_case_preserves_notes_provenance_and_source_inputs(self):
        provenance = {'kind': 'regenerated', 'label': 'Regenerated preview', 'description': 'Rendered from logged poses.'}
        ep = episode(ce={'image_provenance': provenance}, graph={'ce': True, 'nodes': {'a': {'sky': '/api/ceview/test/split/episode/n0'}}})
        note = {'tags': ['stopping', 'reviewed'], 'note': 'Check the goal landmark.'}
        original_ep, original_note = copy.deepcopy(ep), copy.deepcopy(note)
        case = build_case(RUN, ep, note)
        self.assertEqual(case['review'], note)
        self.assertEqual(case['provenance']['images'], provenance)
        self.assertEqual(case['images'][0]['urls'][0], '/api/ceview/test/split/episode/n0_0.jpg')
        self.assertEqual(case['images'][0]['availability'], 'not_checked')
        self.assertEqual(ep, original_ep)
        self.assertEqual(note, original_note)
        later = build_case(RUN, ep, {'note': 'Different review'})
        self.assertEqual(case['provenance']['trace_sha256'], later['provenance']['trace_sha256'])
        case['trace']['frames'][0]['vp'] = 'changed'
        self.assertEqual(ep, original_ep)
        self.assertEvidenceResolves(later)

    def test_unknown_ce_provenance_is_not_reported_as_original(self):
        case = build_case(dict(RUN, env='continuous'), episode())
        self.assertEqual(case['provenance']['images']['kind'], 'unknown')
        self.assertIsNone(case['provenance']['images']['exact_model_input'])

    def test_empty_queue_preserves_category_contract(self):
        queue = build_queue(RUN, [])
        self.assertEqual(queue['total_episodes'], 0)
        self.assertEqual(queue['cases'], [])
        self.assertEqual({c['id'] for c in queue['categories']}, set(CATEGORIES))
        self.assertTrue(all(c['count'] == 0 for c in queue['categories']))

    def test_threshold_edges_and_priority_order_are_deterministic(self):
        good = row(episode(instr_id='clean'))
        good.update(sr=1, osr=1, spl=0.5, stuck=0, ctrl_miss_max=0.999, oscillations=1,
                    first_nonbest_conf=0.8999)
        flagged = dict(good, instr_id='flagged', stuck=1, ctrl_miss_max=1, oscillations=2)
        queue = build_queue(RUN, [good, flagged])
        self.assertEqual([c['instr_id'] for c in queue['cases']], ['flagged'])
        self.assertEqual([c['kind'] for c in queue['cases'][0]['flags']], ['blocked_movement', 'waypoint_miss', 'oscillation'])
        self.assertEqual(queue, build_queue(RUN, [good, flagged]))


if __name__ == '__main__':
    unittest.main()

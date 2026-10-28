"""Regression checks for recorded CE metrics, rendering provenance, and review exports."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from navigation_explorer import ce_runs
from navigation_explorer import parse_trajectories as pt


class RecordedMetricsTests(unittest.TestCase):
    def fixture(self, success=0, oracle=1, near_frames=0, budget=15, scores=None):
        distances = {'n0': 4.0, 'n1': 2.0, 'r0': 0.0}
        graph = SimpleNamespace(pos={v: (i, 0, 0) for i, v in enumerate(distances)}, refs=['r0'],
                                dist=lambda vp, _: distances[vp], euclid=lambda a, b: 1.0,
                                node_dist=lambda a, b: 1.0)
        decisions = [{'at': 'n0', 'target': 'n1', 'action': 'local', 'exec': {'miss_m': .1}},
                     {'at': 'n1', 'target': 'n0', 'action': 'stop', 'exec': {}}]
        rec = {'metrics': {'success': success, 'oracle_success': oracle, 'distance_to_goal': 4.0,
                           'spl': .1, 'ndtw': .2, 'sdtw': .2 * success, 'path_length': 6.0,
                           'collisions': .25, 'steps_taken': 4},
               'max_steps': budget, 'track_goal_dist': [5.0, 2.0, 4.0], 'track': [[0, 0, 0]] * 3,
               'ghosts': {}, 'steps': [{'front': 'n0'}, {'stop_at': 'n0', 'stop_scores': scores or {}}]}
        ep = {'metrics': {}, 'diag': {'near_goal_hops': near_frames, 'closest_hop': 1, 'closest_dist': 4.0},
              'outcome': 'wrong_stop', 'failure_mode': 'never near goal: stopped elsewhere'}
        return ep, rec, graph, decisions

    def test_dense_goal_visit_corrects_label_without_inventing_replay_frame(self):
        ep, rec, graph, decisions = self.fixture()
        ce_runs.CERun._apply_ce_metrics(None, ep, rec, graph, decisions)
        self.assertEqual(ep['failure_mode'], 'passed goal: did not stop there')
        self.assertEqual(ep['metrics']['osr'], 1)
        self.assertTrue(ep['diag']['goal_visit_unsampled'])
        self.assertEqual(ep['diag']['near_goal_hops'], 0)
        self.assertEqual(ep['diag']['closest_hop'], 1)
        self.assertEqual(ep['diag']['closest_dist'], 4.0)
        self.assertEqual(ep['diag']['first_goal_track_index'], 1)
        self.assertEqual(ep['diag']['dense_closest_dist'], 2.0)

    def test_stop_score_label_requires_visited_goal_node_evidence(self):
        ep, rec, graph, decisions = self.fixture(scores={'n0': .8, 'n1': .2})
        ce_runs.CERun._apply_ce_metrics(None, ep, rec, graph, decisions)
        self.assertEqual(ep['failure_mode'], 'passed goal: stop score preferred a wrong node')
        ep, rec, graph, decisions = self.fixture(scores={'n0': .8})
        ce_runs.CERun._apply_ce_metrics(None, ep, rec, graph, decisions)
        self.assertEqual(ep['failure_mode'], 'passed goal: did not stop there')

    def test_recorded_success_and_budget_override_stale_sampled_labels(self):
        for success, oracle, budget, expected, mode in [
            (1, 1, 2, 'success', 'success'),
            (0, 0, 2, 'step_limit', 'never near goal: ran out of steps'),
            (0, 0, 15, 'wrong_stop', 'never near goal: stopped elsewhere'),
        ]:
            with self.subTest(success=success, oracle=oracle, budget=budget):
                ep, rec, graph, decisions = self.fixture(success, oracle, budget=budget)
                ep['failure_mode'] = 'passed goal: did not stop there'
                ce_runs.CERun._apply_ce_metrics(None, ep, rec, graph, decisions)
                self.assertEqual(ep['outcome'], expected)
                self.assertEqual(ep['failure_mode'], mode)

    def test_recorded_success_controls_final_subtask_even_with_sampled_failure(self):
        graph = pt.ScanGraph('test', {'a': (0, 0, 0), 'b': (4, 0, 0), 'c': (8, 0, 0)},
                             {'a': {'b': 4}, 'b': {'a': 4, 'c': 4}, 'c': {'b': 4}},
                             {'a': 0, 'b': 0, 'c': 0}, {}, {'a': 1, 'b': 1, 'c': 1})
        gt = {'scan': 'test', 'path': ['a', 'b', 'c'], 'instruction': 'Go there.'}
        sub = {'instruction': 'Go there.', 'subs': [['Go', 'there.']], 'chunks': [[0, 3]]}
        with patch.object(graph, 'arrival_pose', return_value=None):
            ep = pt.analyse_episode({'instr_id': 'one', 'trajectory': [['a'], ['b']]}, gt, graph,
                                    subtasks=sub, success_for_subtasks=True)
            self.assertEqual(ep['metrics']['sr'], 0)
            self.assertEqual(ep['subtasks']['reached'], [1])
            self.assertEqual(ep['diag']['subtasks_done'], 1)
            summary = pt.analyse_episode({'instr_id': 'one', 'trajectory': [['a'], ['b']]}, gt, graph,
                                         subtasks=sub, success_for_subtasks=True, full=False)
            self.assertEqual(summary['diag']['subtasks_done'], 1)


class FullExportTests(unittest.TestCase):
    def test_full_exports_keep_reviews_without_overwriting_recorded_evidence(self):
        evidence = {'instr_id': 'one', 'instruction': 'Recorded instruction', 'metrics': {'sr': 1},
                    'frames': [{'decision': 0}], 'decisions': [{'t': 0}]}
        run = SimpleNamespace(name='run', pred_file='preds', anno_file='annos',
                              episode=lambda iid, full: evidence)
        row = {'instr_id': 'one', 'instruction': 'Stale summary', 'metrics': {'sr': 0},
               'tags': ['reviewed', 'stop'], 'note': 'Inspect decision zero.'}
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ('jsonl', 'json'):
                with self.subTest(format=fmt):
                    path = pt.write_export(run, [row], Path(tmp) / ('case.' + fmt), fmt, full=True)
                    exported = json.loads(path.read_text())
                    if fmt == 'json':
                        exported = exported['episodes'][0]
                    self.assertEqual(exported['tags'], row['tags'])
                    self.assertEqual(exported['note'], row['note'])
                    self.assertEqual(exported['metrics']['sr'], 1)
                    self.assertEqual(exported['instruction'], 'Recorded instruction')
                    self.assertEqual(exported['decisions'], evidence['decisions'])
        self.assertNotIn('tags', evidence)

    def test_cli_without_reviews_does_not_invent_them(self):
        run = SimpleNamespace(episode=lambda iid, full: {'instr_id': iid})
        with tempfile.TemporaryDirectory() as tmp:
            path = pt.write_export(run, [{'instr_id': 'one'}], Path(tmp) / 'case.jsonl', 'jsonl', full=True)
            self.assertEqual(json.loads(path.read_text()), {'instr_id': 'one'})


class ImageProvenanceTests(unittest.TestCase):
    def test_receipt_applies_only_to_unchanged_covered_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'logs' / 'episodes_val_unseen.jsonl'
            log.parent.mkdir()
            log.write_text('{"episode_id":"one"}\n')
            migration = root / 'migration'
            migration.mkdir()
            status = {'status': 'complete', 'input_logs': [
                {'path': log.relative_to(root).as_posix(), 'sha256': hashlib.sha256(log.read_bytes()).hexdigest()}]}
            (migration / 'local_ce_cube_views_status.json').write_text(json.dumps(status))
            receipt = {'status': 'pass', 'errors': [], 'provenance': {
                'kind': 'regenerated_preview', 'pose_source': 'Rounded logged poses', 'runtime': 'test renderer'}}
            (migration / 'local_ce_cube_views_validation.json').write_text(json.dumps(receipt))
            provenance = ce_runs.image_provenance(log, root)
            self.assertEqual(provenance['kind'], 'regenerated_preview')
            self.assertEqual(provenance['runtime'], 'test renderer')
            self.assertIn('not the original', provenance['description'])
            log.write_text('{"episode_id":"changed"}\n')
            self.assertEqual(ce_runs.image_provenance(log, root)['kind'], 'unknown')
            other = log.with_name('episodes_other.jsonl')
            other.write_text('{"episode_id":"one"}\n')
            self.assertEqual(ce_runs.image_provenance(other, root)['kind'], 'unknown')

    def test_missing_or_invalid_receipt_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'episodes.jsonl'
            log.write_text('{}\n')
            self.assertEqual(ce_runs.image_provenance(log, root)['kind'], 'unknown')
            (root / 'migration').mkdir()
            (root / 'migration/local_ce_cube_views_status.json').write_text('broken json')
            (root / 'migration/local_ce_cube_views_validation.json').write_text('{}')
            self.assertEqual(ce_runs.image_provenance(log, root)['kind'], 'unknown')


if __name__ == '__main__':
    unittest.main()

"""HTTP regressions for agent review queries; no datasets, GPU, or note writes required."""
import copy
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from navigation_explorer.server import App, Handler


class InvestigationAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [
            {'instr_id': 'failed', 'scan': 'building-a', 'instruction': 'Go to the blue door',
             'outcome': 'wrong_stop', 'sr': 0, 'osr': 1, 'ne': 5, 'spl': 0, 'tl': 12,
             'gt_len': 6, 'steplog': True, 'near_goal_hops': 1},
            {'instr_id': 'detour', 'scan': 'building-b', 'instruction': 'Walk to the blue door',
             'outcome': 'success', 'sr': 1, 'osr': 1, 'ne': 1, 'spl': .2, 'tl': 30,
             'gt_len': 6, 'steplog': True, 'oscillations': 3, 'near_goal_hops': 1},
            {'instr_id': 'ordinary', 'scan': 'building-c', 'instruction': 'Stop at the chair',
             'outcome': 'success', 'sr': 1, 'osr': 1, 'ne': 0, 'spl': 1, 'tl': 4,
             'gt_len': 4, 'steplog': True, 'near_goal_hops': 1},
        ]
        cls.notes = {'failed': {'tags': ['reviewed'], 'note': 'Check the stopping decision.'}}
        run = SimpleNamespace(model='example', split='val_unseen', pred_file='missing-predictions.json',
                              anno_file='missing-annotations.json', steplog_file=None,
                              route_reference=False, max_action_len=15, ids=[r['instr_id'] for r in cls.rows],
                              settings={}, rows=lambda: cls.rows)
        cls.app = App.__new__(App)
        cls.app.runs = {'example/val_unseen': run}
        cls.app.notes = SimpleNamespace(for_run=lambda name: {'notes': cls.notes, 'vocabulary': ['reviewed']})

        def episode(name, instr_id):
            row = next(r for r in cls.rows if r['instr_id'] == instr_id)
            return {'instr_id': instr_id, 'scan': row['scan'], 'instruction': row['instruction'],
                    'outcome': row['outcome'], 'failure_mode': 'passed goal: did not stop there',
                    'metrics': {k: row[k] for k in ('sr', 'osr', 'ne', 'spl', 'tl', 'gt_len')},
                    'diag': {'near_goal_hops': row['near_goal_hops'], 'steplog': True},
                    'frames': [{'vp': 'a', 'decision': 0, 'dist': 0}, {'vp': 'b', 'decision': 1, 'dist': 5}],
                    'decisions': [{'t': 0, 'at': 'a', 'target': 'b', 'action': 'local', 'route': ['b'],
                                   'model': {'fused': [.1, .9]},
                                   'ref': {'in_area': True, 'stop': .1, 'conf': .9, 'ok': False, 'type': 'missed stop'}},
                                  {'t': 1, 'at': 'b', 'target': 'b', 'action': 'stop', 'route': []}],
                    'path': ['a', 'b'], 'gt_path': ['a'], 'goal': 'a', 'start': 'a', 'final': 'b',
                    'goal_zone': ['a'], 'dist_to_goal': {'a': 0, 'b': 5}, 'subtasks': {}}
        cls.app.episode = episode
        handler = type('TestHandler', (Handler,), {'app': cls.app})
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, path='/api/investigations', **query):
        query.setdefault('run', 'example/val_unseen')
        with urlopen(self.base + path + '?' + urlencode(query), timeout=5) as response:
            return json.load(response), response.headers

    def test_queue_includes_successes_and_paginates_without_duplicates(self):
        full, _ = self.get(limit=200)
        self.assertEqual(full['total_episodes'], 3)
        self.assertIn('detour', {c['instr_id'] for c in full['cases']})
        first, _ = self.get(limit=1)
        second, _ = self.get(limit=1, offset=1)
        self.assertEqual(first['total_matches'], full['total_matches'])
        self.assertNotEqual(first['cases'][0]['instr_id'], second['cases'][0]['instr_id'])
        beyond, _ = self.get(offset=100)
        self.assertEqual(beyond['cases'], [])

    def test_filters_intersect_and_keep_global_category_counts(self):
        full, _ = self.get()
        selected, _ = self.get(outcome='success', q='BLUE DOOR')
        self.assertEqual([c['instr_id'] for c in selected['cases']], ['detour'])
        self.assertEqual(selected['categories'], full['categories'])
        building, _ = self.get(q='building-a', kind='missed_stop')
        self.assertEqual([c['instr_id'] for c in building['cases']], ['failed'])

    def test_bad_queries_and_unknown_run_return_json_errors(self):
        for query in ({'limit': 0}, {'limit': 201}, {'offset': -1}, {'offset': 'NaN'},
                      {'kind': 'invented'}, {'outcome': 'invented'}, {'run': 'unknown'}):
            with self.subTest(query=query), self.assertRaises(HTTPError) as ctx:
                self.get(**query)
            self.assertEqual(ctx.exception.code, 404 if 'run' in query else 400)
            self.assertIn('error', json.load(ctx.exception))
            ctx.exception.close()

    def test_case_download_preserves_review_and_records_source_availability(self):
        before = copy.deepcopy(self.notes)
        case, headers = self.get('/api/investigation', id='failed', download=1)
        self.assertEqual(case['schema_version'], 'navigation-investigation/v1')
        self.assertEqual(case['review']['tags'], ['reviewed'])
        self.assertEqual(case['review']['note'], before['failed']['note'])
        self.assertIn('attachment;', headers['Content-Disposition'])
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertFalse(any(s['available'] for s in case['run']['sources']))
        self.assertIn('run=example%2Fval_unseen', case['api_url'])
        self.assertEqual(self.notes, before)

    def test_unknown_case_is_404(self):
        with self.assertRaises(HTTPError) as ctx:
            self.get('/api/investigation', id='absent')
        self.assertEqual(ctx.exception.code, 404)
        ctx.exception.close()


if __name__ == '__main__':
    unittest.main()

"""Portable installation and API contracts using only bundled synthetic recordings."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import navigation_explorer
from navigation_explorer.configuration import load_project
from navigation_explorer.server import App, Notes, compatible, create_server

DEMO = Path(navigation_explorer.__file__).parent / 'demo' / 'project.json'


class ProjectHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.state = Path(cls.tmp.name) / 'review-state'
        cls.app = App(DEMO, cls.state)
        cls.server = create_server(cls.app, port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def get(self, path, **query):
        with urlopen(self.base + path + ('?' + urlencode(query) if query else ''), timeout=5) as response:
            return json.load(response)

    def post(self, path, value, headers=None):
        request = Request(self.base + path, json.dumps(value).encode(),
                          {'Content-Type': 'application/json', **(headers or {})})
        with urlopen(request, timeout=5) as response:
            return response.read(), response.headers

    def test_bundled_demo_contains_both_environments_and_unknown_evidence(self):
        meta = self.get('/api/meta')
        self.assertEqual({run['env'] for run in meta['runs']}, {'discrete', 'continuous'})
        self.assertTrue(all(not run['capabilities']['whatif'] for run in meta['runs']))
        unscored = next(run for run in meta['runs'] if run['name'] == 'demo/unscored')
        self.assertNotIn('sr', unscored['summary'])
        rows = self.get('/api/rows', run='demo/unscored')
        self.assertEqual(rows[0]['outcome'], 'unscored')
        episode = self.get('/api/episode', run='demo/unscored', id=rows[0]['instr_id'])
        self.assertFalse(episode['capabilities']['images'])
        self.assertFalse(episode['capabilities']['mesh'])
        self.assertFalse(episode['capabilities']['decisions'])
        self.assertNotEqual(episode['decisions'][-1]['action'], 'stop')

    def test_all_cases_are_strict_json_with_resolvable_evidence(self):
        for name, run in self.app.runs.items():
            for instr_id in run.ids:
                with self.subTest(run=name, instr_id=instr_id):
                    case = self.get('/api/investigation', run=name, id=instr_id)
                    json.dumps(case, allow_nan=False)
                    for finding in case['findings']:
                        for fact in finding['evidence']:
                            value = case
                            for part in fact['path'].lstrip('/').split('/'):
                                part = part.replace('~1', '/').replace('~0', '~')
                                value = value[int(part)] if isinstance(value, list) else value[part]
                            self.assertEqual(value, fact['value'])

    def test_notes_survive_reload_and_full_exports(self):
        run, instr_id = 'demo/discrete', 'graph-missed-stop'
        body, note_headers = self.post('/api/notes', {'run': run, 'id': instr_id, 'tags': ['reviewed'], 'note': 'Recheck stop timing.'})
        self.assertEqual(json.loads(body)['tags'], ['reviewed'])
        self.assertEqual(note_headers['Cache-Control'], 'no-store')
        stored = Notes(self.state / 'episode_notes.json').for_run(run)['notes'][instr_id]
        self.assertEqual(stored['note'], 'Recheck stop timing.')
        body, headers = self.post('/api/export', {'run': run, 'ids': [instr_id], 'format': 'jsonl', 'full': True})
        exported = json.loads(body)
        self.assertEqual(exported['note'], stored['note'])
        self.assertIn('reviewed', exported['tags'])
        self.assertIn('frames', exported)
        self.assertIn('attachment', headers['Content-Disposition'])

    def test_invalid_filters_and_missing_native_assets_return_json_errors(self):
        for path, query, code in [
            ('/api/investigations', {'run': 'demo/discrete', 'limit': 'bad'}, 400),
            ('/api/investigation', {'run': 'demo/discrete', 'id': 'absent'}, 404),
            ('/api/scan', {'run': 'demo/discrete', 'scan': '../private'}, 400),
            ('/api/legacy-view/demo%2Fdiscrete/room/node_0.jpg', {}, 404),
            ('/static/%2e%2e/server.py', {}, 404),
        ]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as ctx:
                self.get(path, **query)
            self.assertEqual(ctx.exception.code, code)
            self.assertIn('error', json.load(ctx.exception))
            ctx.exception.close()

    def test_cross_origin_notes_and_unconfigured_inference_are_rejected(self):
        for path, body, headers, code in [
            ('/api/notes', {'run': 'demo/discrete', 'id': 'graph-missed-stop'}, {'Origin': 'https://another.example'}, 403),
            ('/api/whatif/run', {'run': 'demo/discrete'}, {}, 409),
            ('/api/notes', [], {}, 400),
        ]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as ctx:
                self.post(path, body, headers)
            self.assertEqual(ctx.exception.code, code)
            ctx.exception.close()


class ConfigurationTests(unittest.TestCase):
    def manifest(self):
        data = json.loads(DEMO.read_text())
        for run in data['runs']:
            run['episodes'] = str(DEMO.parent / run['episodes'])
        return data

    def test_duplicate_identity_unknown_adapter_and_environment_mismatch_fail(self):
        for change in ('duplicate', 'adapter', 'environment'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                data = self.manifest()
                if change == 'duplicate':
                    data['runs'][1]['id'] = data['runs'][0]['id']
                elif change == 'adapter':
                    data['runs'][0]['adapter'] = 'exec-custom-python'
                else:
                    data['runs'][0].update(adapter='r2r', environment='continuous')
                path = Path(temp) / 'project.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_project(path)

    def test_relative_paths_do_not_depend_on_process_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, PYTHONPATH=str(Path(navigation_explorer.__file__).parent.parent))
            result = subprocess.run([sys.executable, '-m', 'navigation_explorer', '--project', str(DEMO),
                                     '--check', '--state-dir', str(Path(temp) / 'state')],
                                    cwd=temp, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Validated 3 runs, 4 episodes', result.stdout)

    def test_unrelated_datasets_do_not_compare_and_compared_nodes_are_namespaced(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self.manifest()
            original = data['runs'][0]
            aligned = dict(original, id='second-agent', model='Second model')
            unrelated = dict(original, id='other-dataset', dataset='Different dataset')
            data['runs'] = [original, aligned, unrelated]
            path = Path(temp) / 'project.json'
            path.write_text(json.dumps(data))
            app = App(path, Path(temp) / 'state')
            first = app.meta()['runs'][0]
            self.assertEqual(first['comparable_runs'], ['second-agent'])
            ep = app.episode(original['id'], 'graph-missed-stop')
            self.assertEqual(list(ep['others']), ['second-agent'])
            compared = ep['others']['second-agent']
            self.assertTrue(all(node.startswith('cmp:second-agent:') for node in compared['path']))
            self.assertTrue(all(node in compared['nodes'] for node in compared['path']))

    def test_explicit_comparison_groups_do_not_fall_back_to_matching_dataset_labels(self):
        base = dict(dataset='Room dataset', split='validation', ce=False, coordinate_system='cartesian_z_up_m')
        a = SimpleNamespace(name='a', comparison_group='study-a', **base)
        for group in ('study-b', None):
            with self.subTest(group=group):
                self.assertFalse(compatible(a, SimpleNamespace(name='b', comparison_group=group, **base)))
        self.assertTrue(compatible(a, SimpleNamespace(name='b', comparison_group='study-a', **base)))

    def test_mesh_conversion_writes_only_to_runtime_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'scans' / 'scene' / 'house_segmentations'
            source.mkdir(parents=True)
            (source / 'scene.ply').write_bytes(b'source fixture')
            external = root / 'read-only-assets'
            external.mkdir()
            app = App.__new__(App)
            app.runs = {'run': SimpleNamespace(scans_dir=root / 'scans')}
            app.assets = {'run': {'meshes': external}}
            generated = root / 'state' / 'meshes'
            app.generated_assets = {'run': {'meshes': generated}}
            app.mesh_lock, app.mesh_building = threading.Lock(), {}

            def convert(scan, scans, output):
                (Path(output) / f'{scan}.bin.gz').write_bytes(b'generated mesh')

            converter = SimpleNamespace(build=convert)
            with patch.dict(sys.modules, {'navigation_explorer.build_meshes': converter}):
                self.assertEqual(app.mesh('run', 'scene'), b'generated mesh')
            self.assertEqual(list(external.iterdir()), [])
            self.assertTrue((generated / 'scene.bin.gz').is_file())


if __name__ == '__main__':
    unittest.main()

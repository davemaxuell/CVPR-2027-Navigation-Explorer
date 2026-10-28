"""Command line entry point for source checkouts and installed wheels."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__
from .server import App, create_server, json_safe


def main(argv=None):
    parser = argparse.ArgumentParser(description='Replay and investigate recorded navigation agents.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--project', type=Path, help='Project JSON manifest; paths resolve relative to this file')
    source.add_argument('--demo', action='store_true', help='Open bundled synthetic discrete and continuous examples (default)')
    parser.add_argument('--state-dir', type=Path, help='Writable directory for local review notes and caches')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (default: localhost)')
    parser.add_argument('--port', type=int, default=8765, help='HTTP port; 0 chooses an available port')
    parser.add_argument('--open-browser', action='store_true', help='Open the viewer in the local desktop browser')
    parser.add_argument('--check', action='store_true', help='Validate and load all configured runs, then exit')
    parser.add_argument('--version', action='version', version=f'Navigation Explorer {__version__}')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    project = args.project or Path(__file__).parent / 'demo' / 'project.json'
    try:
        app = App(project, args.state_dir)
        if args.check:
            # Validate full episode playback, not only cached summary rows.
            for run in app.runs.values():
                for instr_id in run.ids:
                    json.dumps(json_safe(run.episode(instr_id)), allow_nan=False)
            print(f'Validated {len(app.runs)} runs, {sum(len(run.ids) for run in app.runs.values())} episodes.')
            return 0
        server = create_server(app, args.host, args.port)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Navigation Explorer: {error}', file=sys.stderr)
        return 2
    address = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    url = f'http://{address}:{server.server_port}/'
    print(f'Navigation Explorer on {url}', flush=True)
    print(f'Local reviews and caches: {app.state_dir}', flush=True)
    print('Press Ctrl+C to stop.', flush=True)
    if args.open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

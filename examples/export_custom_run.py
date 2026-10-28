"""Export a tiny synthetic run through the same interface a model logger uses."""
import argparse
import json
from pathlib import Path

from navigation_explorer.writer import episode_from_positions, write_recordings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='directory for project.json and episodes.jsonl')
    parser.add_argument('--environment', choices=('discrete', 'continuous'), default='continuous')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    project = args.output / 'project.json'
    if project.exists() and not args.overwrite:
        parser.error(f'{project} already exists; use a new output directory or --overwrite')

    # Replace these positions with your logger's actual samples, in z-up meters.
    # For Habitat [x,y,z] positions, convert to [x,-z,y] first. Add camera height
    # only when the supplied images/poses describe a camera above the agent base.
    positions = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]]
    episode = episode_from_positions(
        instr_id='example-001', scene='example-room', instruction='Walk around the corner.',
        positions=positions,
        node_ids=['start', 'hall', 'corner', 'end'] if args.environment == 'discrete' else None,
        metadata={'synthetic': True, 'description': 'Replace with your model recording.'},
        # Supply metrics and decisions from your evaluator/logger when available.
        # Omitted evaluation remains unscored; an end position does not imply STOP.
    )
    write_recordings(args.output / 'episodes.jsonl', [episode], environment=args.environment,
                     overwrite=args.overwrite)
    manifest = {'schema_version': 'navigation-explorer/project-v1', 'name': 'My navigation recordings',
                'runs': [{'id': 'my-model/inspection', 'adapter': 'recording', 'model': 'My navigator',
                          'dataset': 'My dataset', 'split': 'inspection', 'environment': args.environment,
                          'episodes': 'episodes.jsonl'}]}
    project.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {project}')
    print(f'Run: navigation-explorer --project "{project}"')


if __name__ == '__main__':
    main()

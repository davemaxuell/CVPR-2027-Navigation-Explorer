"""Sub-instructions for RxR, aligned to the ground-truth path by the guide's own timing.

  python -m navigation_explorer.rxr_subtasks --rxr-dir /path/to/RxR --out /path/to/subtasks                      # English val_seen + val_unseen
  python -m navigation_explorer.rxr_subtasks --rxr-dir /path/to/RxR --out /path/to/subtasks --splits val_unseen --languages en-US en-IN hi-IN

RxR has no sub-instruction annotation like FGR2R, but every guide recorded the instruction while walking
the path: `timed_instruction` gives each spoken word a time, and the guide's pose trace gives the panorama
the guide stood at over time. Each instruction is split into steps (sentences, and comma/"and"/"then"
clauses that start a new action such as "turn", "go", "walk"). A step's sub-goal is the path viewpoint
where the guide stood when they started speaking the next step; the last step's sub-goal is the goal.

Output: cache/rxr_subtasks/rxr_<split>.json, {instruction_id: {'instruction', 'spans': [[start, end), ...]
character spans, 'chunks': [[s, e], ...] 1-indexed path ranges as in FGR2R's chunk_view}}. The viewer
loads it when the run manifest supplies the subtasks file. Needs numpy (pose traces are .npz).
"""
import argparse
import difflib
import gzip
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


# words that start a new navigation step after a comma, "and" or "then"
STEP_WORDS = {'turn', 'go', 'walk', 'head', 'enter', 'exit', 'leave', 'take', 'continue', 'move', 'step', 'stop',
              'climb', 'proceed', 'keep', 'pass', 'follow', 'cross', 'wait', 'stand', 'make', 'come', 'get',
              'descend', 'ascend', 'approach', 'veer', 'bear', 'pivot', 'once', 'after', 'when', 'finally', 'now'}
_BREAK = re.compile(r"[.!?;]+\s+|,\s+(?=(?:(?:and|then|and then|so)\s+)?(\w+))|\s+(?=(?:and then|then)\s+(\w+))", re.I)
_TOKEN = re.compile(r'\S+')
MIN_WORDS = 3          # shorter pieces are merged into the previous step


def split_steps(text):
    """Character spans [start, end) of the navigation steps in an instruction."""
    cuts = [0]
    for m in _BREAK.finditer(text):
        follower = (m.group(1) or m.group(2) or '').lower()
        sentence_end = m.group().strip()[:1] in '.!?;'
        if (sentence_end or follower in STEP_WORDS) and m.end() < len(text.rstrip()):
            cuts.append(m.end())
    spans = [[s, e] for s, e in zip(cuts, cuts[1:] + [len(text)])]
    merged = []
    for s, e in spans:
        if merged and len(text[s:e].split()) < MIN_WORDS:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if len(merged) > 1 and len(text[merged[0][0]:merged[0][1]].split()) < MIN_WORDS:
        first = merged.pop(0)
        merged[0][0] = first[0]
    return merged


def word_times(text, timed):
    """Start time of each whitespace word of `text` (None where the timed transcript has no match/time),
    and the fraction of words matched. The final text is an edited version of the timed transcript."""
    norm = lambda w: ''.join(c for c in w.lower() if c.isalnum())
    words = [(m.start(), norm(m.group())) for m in _TOKEN.finditer(text)]
    spoken = [(norm(w.get('word', '')), w.get('start_time')) for w in timed]
    sm = difflib.SequenceMatcher(None, [w for w, _ in spoken], [w for _, w in words], autojunk=False)
    times, matched = [None] * len(words), 0
    for a, b, size in sm.get_matching_blocks():
        matched += size
        for k in range(size):
            t = spoken[a + k][1]
            if isinstance(t, (int, float)) and math.isfinite(t):
                times[b + k] = float(t)
    return [pos for pos, _ in words], times, matched / max(1, len(words))


def align(item, trace):
    """Steps of one instruction with their path chunks, or None when the timing cannot be trusted."""
    text, path = item['instruction'], item['path']
    starts, times, matched = word_times(text, item.get('timed_instruction') or [])
    if matched < 0.6 or len(path) < 2:
        return None
    index = {vp: i for i, vp in enumerate(path)}
    panos, ptimes = trace['pano'], trace['time']

    def path_index_at(t):
        j = int(np.searchsorted(ptimes, t, side='right')) - 1
        for k in range(max(j, 0), -1, -1):          # latest sample at or before t that is on the path
            if panos[k] in index:
                return index[panos[k]]
        return 0

    spans = split_steps(text)
    ends, prev = [], 0
    for k, (s, e) in enumerate(spans):
        if k == len(spans) - 1:
            ends.append(len(path) - 1)
            break
        nxt = spans[k + 1][0]
        t = next((times[i] for i, p in enumerate(starts) if p >= nxt and times[i] is not None), None)
        prev = max(prev, path_index_at(t) if t is not None else prev)
        ends.append(prev)
    chunks, s = [], 1
    for e in ends:
        chunks.append([s, e + 1])
        s = e + 1
    return {'instruction': text, 'spans': spans, 'chunks': chunks}


def build(split, languages, rxr_dir, out_dir):
    rxr_dir, out_dir = Path(rxr_dir), Path(out_dir)
    out, skipped = {}, 0
    with gzip.open(rxr_dir / f'rxr_{split}_guide.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if languages and item['language'] not in languages:
                continue
            trace = rxr_dir / 'pose_traces' / f'rxr_{split}' / f"{item['instruction_id']:06d}_guide_pose_trace.npz"
            res = align(item, np.load(trace)) if trace.is_file() else None
            if res is None:
                skipped += 1
                continue
            out[str(item['instruction_id'])] = res
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f'rxr_{split}.json'
    dest.write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    return dest, len(out), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rxr-dir', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--splits', nargs='+', default=['val_seen', 'val_unseen'])
    ap.add_argument('--languages', nargs='+', default=['en-US', 'en-IN'])
    args = ap.parse_args()
    for split in args.splits:
        dest, n, skipped = build(split, set(args.languages), args.rxr_dir, args.out)
        print(f'{split}: {n} instructions aligned, {skipped} without usable timing -> {dest}')


if __name__ == '__main__':
    main()

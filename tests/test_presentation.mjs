import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../src/navigation_explorer/static/presentation.js', import.meta.url), 'utf8');
const { episodeSummary, episodeEvents } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const ep = {
  goal: 'goal', ce: {}, outcome: 'wrong_stop', metrics: { ne: 5.5, osr: 1 },
  diag: { closest_hop: 1, closest_dist: 3.1, goal_visit_unsampled: true, first_goal_track_index: 94 },
  frames: [{ vp: 'start', decision: 0 }, { vp: 'end', decision: 1 }],
  decisions: [{ action: 'local' }, { action: 'stop' }],
};
assert.match(episodeSummary(ep, 3, 'frame'), /full trajectory entered the success area/);
assert.match(episodeSummary(ep, 3, 'frame'), /Closest sampled approach: 3.10 m at frame 1/);
const event = episodeEvents(ep, 3).find(e => e.id === 'closest');
assert.equal(event.frame, 1, 'Dense track indices must never become replay frame indices');
assert.equal(event.label, 'Closest sampled approach');
assert.match(event.detail, /absent from these replay frames/);
const discrete = { ...ep, ce: undefined, diag: { closest_hop: 1, closest_dist: 0 } };
assert.match(episodeSummary(discrete, 3), /^Within 3 m at hop 1/);
assert.equal(episodeEvents(discrete, 3)[0].label, 'Closest approach');
assert.match(episodeSummary({ goal: null }, 3), /No reference goal/);
console.log('Presentation regressions passed: unsampled visit, frame bounds, ordinary replay, no ground truth.');

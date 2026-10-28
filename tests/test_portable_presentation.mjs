import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const source = await readFile(new URL('../src/navigation_explorer/static/presentation.js', import.meta.url), 'utf8');
const {successRadius, datasetTitle, comparableRun, finalActionLabel, episodeEvents, episodeSummary} =
  await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

assert.equal(successRadius({evaluation: {success_radius_m: null}}, 3), null);
assert.equal(successRadius({evaluation: {success_radius_m: 0.4}}, 3), 0.4);
assert.equal(successRadius({evaluation: {success_radius_m: -1}}, 3), null);
assert.equal(successRadius({evaluation: {success_radius_m: 3, distance_kind: 'unknown'}}, 3), null);
assert.equal(datasetTitle({dataset: 'Warehouse robot trials', env: 'continuous'}), 'Warehouse robot trials');
assert.equal(datasetTitle({}), 'Custom dataset');
const run = {name: 'model-a/eval', dataset: 'Warehouse', split: 'eval', env: 'continuous', coordinate_system: 'building-A'};
assert.equal(comparableRun(run, {...run, name: 'model-b/eval'}), true);
assert.equal(comparableRun(run, {...run, name: 'model-b/eval', coordinate_system: 'building-B'}), false);
assert.equal(comparableRun(run, {...run, name: 'model-b/eval', dataset: 'Different data'}), false);
assert.equal(comparableRun({...run, comparable_runs: []}, {...run, name: 'model-b/eval'}), false);
assert.equal(comparableRun({...run, comparable_runs: ['paired']}, {name: 'paired'}), true);

const unscored = {goal: 'goal', outcome: 'unscored', evaluation: {success_radius_m: null}, metrics: {}, diag: {},
  frames: [{vp: 'a', decision: 0}, {vp: 'b', decision: 0}], decisions: [{action: 'unknown', source: 'replay_placeholder'}]};
assert.equal(finalActionLabel(unscored), 'END');
assert.match(episodeSummary(unscored, null), /Outcome not evaluated/);
assert.doesNotMatch(episodeSummary(unscored, null), /stopped|3 m|failed/);
assert.equal(episodeEvents(unscored, null).at(-1).label, 'Episode end');
const failed = {...unscored, outcome: 'failure', metrics: {sr: 0, ne: 4}};
assert.match(episodeSummary(failed, null), /recorded episode ended/);
assert.doesNotMatch(episodeSummary(failed, null), /agent stopped|radius/);
assert.equal(finalActionLabel({...unscored, decisions: [{action: 'stop'}]}), 'STOP');
console.log('Portable presentation checks passed: missing evaluation, explicit end actions, metadata, and comparison identity.');

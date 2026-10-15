"""Score a finished run. A separate program: it talks to the server like any client, but the task script never sees it.

    python client/bench/run.py client/bench/pick_place.yaml --out runs/<this run's folder>      # writes <folder>/score.json

--out is required on purpose: the score belongs next to that run's own logs, so nothing has to be matched by timestamp.
From a bigger client, call it directly with your own open connection and save the dict wherever the run keeps its logs:

    from client.bench.run import score
    result = score(yaml.safe_load(open('client/bench/pick_place.yaml')), world)      # world = your WorldController

Rule:  body: <free body>   near: [x, y, z] | <other body>   within: <metres>
Measures the world's true state: reads it, then again `hold` sim-seconds later, and every rule must pass both times.
"""
import argparse
import json
import math
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from world_controller import Primitive, WorldController   # the only thing it takes: the file next to this folder


def score(task: dict, world) -> dict:
    """Check the task's rules against the running server. No printing, no files: returns the record."""
    state = lambda: world.act(Primitive('get_state'))
    snaps = [state()]
    if task.get('hold'):
        while (later := state())['time'] < snaps[0]['time'] + task['hold']:
            time.sleep(0.05)
        snaps.append(later)

    def dist(rule, bodies):
        target = bodies[rule['near']]['position'] if isinstance(rule['near'], str) else rule['near']
        return math.dist(bodies[rule['body']]['position'], target)

    rules = []
    for r in task['rules']:
        d = [dist(r, s['bodies']) for s in snaps]
        rules.append({'rule': r['name'], 'within': r['within'], 'distances': d, 'passed': all(x <= r['within'] for x in d)})
    return {'scene': task.get('scene'), 'passed': all(r['passed'] for r in rules), 'sim_time': snaps[-1]['time'],
            'rules': rules, 'bodies': snaps[-1]['bodies']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task')
    ap.add_argument('--out', required=True, help="the run's folder; score.json is written inside it")
    ap.add_argument('--port', type=int, default=8988)
    args = ap.parse_args()
    task = yaml.safe_load(open(args.task))
    with WorldController(port=args.port) as world:
        result = score(task, world)
    result = {'task': os.path.splitext(os.path.basename(args.task))[0], 'run_dir': os.path.abspath(args.out), **result}
    for r in result['rules']:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['rule']}: {' then '.join(f'{d * 1000:.1f} mm' for d in r['distances'])} (allowed {r['within'] * 1000:g} mm)")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'score.json')
    json.dump(result, open(path, 'w'), indent=1)
    print(f"{'PASSED' if result['passed'] else 'FAILED'}  ({sum(r['passed'] for r in result['rules'])}/{len(result['rules'])} rules)  -> {path}")
    sys.exit(0 if result['passed'] else 1)

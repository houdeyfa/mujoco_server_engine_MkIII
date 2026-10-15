"""Fake task client: exercises every verb the server advertises, plus the error paths.

    python server.py config/ur10e_robotiq.yaml          # terminal 1 (any config with a robot)
    python tests/fake_running_client.py [--port N]      # terminal 2

Exit code 0 = all checks passed. Add a check here whenever you add a verb.
"""
import argparse
import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from client.world_controller import ActionError, Primitive, WorldController

results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ''))


def raises(primitive, world, status):
    try:
        world.act(primitive)
    except ActionError as exc:
        return exc.reply['status'] == status
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8988)
    args = ap.parse_args()

    with WorldController(args.host, args.port, timeout=30) as world:
        robots0 = world.act(Primitive('list_robots'))['robots']
        initial = {'bodies': world.act(Primitive('get_state'))['bodies'],
                   'joints': {r['robot_id']: world.act(Primitive('get_state', r['robot_id']))['joint_positions'] for r in robots0}}
        print('list_primitives')
        prims = world.act(Primitive('list_primitives'))['primitives']
        verbs = {p['verb'] for p in prims}
        check('advertises the verbs it implements', {'list_primitives', 'list_robots', 'get_state'} <= verbs, verbs)
        check('every verb has description + args', all('description' in p and 'args' in p for p in prims))

        print('list_robots')
        robots = world.act(Primitive('list_robots'))['robots']
        check('returns a list of robots with ids', all('robot_id' in r for r in robots), robots)

        print('get_state (robots)')
        for r in robots:
            s = world.act(Primitive('get_state', r['robot_id']))
            check(f"{r['robot_id']}: {r['n_joints']} joint positions", len(s['joint_positions']) == r['n_joints'])
            check(f"{r['robot_id']}: velocities match positions", len(s['joint_velocities']) == r['n_joints'])
            ee = s['end_effector']
            check(f"{r['robot_id']}: end-effector pose is 3 + 4 floats",
                  ee is not None and len(ee['position']) == 3 and len(ee['orientation']) == 4)
            if ee:
                n = sum(x * x for x in ee['orientation']) ** 0.5
                check(f"{r['robot_id']}: orientation is a unit quaternion", abs(n - 1) < 1e-3, n)

        print('cameras')
        cams = world.act(Primitive('list_cameras'))['cameras']
        check('list_cameras returns a list', isinstance(cams, list), cams)
        check('verbs advertised: list_cameras + capture_image', {'list_cameras', 'capture_image'} <= verbs)
        check('no camera pose is ever sent', all('pos' not in c and 'pose' not in c and 'quat' not in c for c in cams))
        for cam in cams[:1]:
            res = world.act(Primitive('capture_image', camera=cam['name']))
            img = res['rgb']
            check(f"{cam['name']}: png decodes to (h, w, 3) uint8",
                  img.shape == (cam['height'], cam['width'], 3) and str(img.dtype) == 'uint8', getattr(img, 'shape', img))
            check(f"{cam['name']}: image is not blank", len({tuple(p) for p in img[::16, ::16].reshape(-1, 3)}) > 3)
            check(f"{cam['name']}: intrinsics have fx fy cx cy", all(k in res['intrinsics'] for k in ('fx', 'fy', 'cx', 'cy')))
            raw = world.act(Primitive('capture_image', camera=cam['name'], encoding='raw'))['rgb']
            check(f"{cam['name']}: raw has the same shape", raw.shape == img.shape)
            dres = world.act(Primitive('capture_image', camera=cam['name'], depth=True))
            depth, rng = dres['depth'], dres['depth_range']
            check(f"{cam['name']}: depth is float32 (h, w)", str(depth.dtype) == 'float32' and depth.shape == (cam['height'], cam['width']), (depth.dtype, depth.shape))
            check(f"{cam['name']}: depth is finite and inside the clip range",
                  bool((depth > 0).all()) and bool((depth <= rng['far'] * 1.001).all()) and bool(depth.min() >= rng['near'] * 0.99), (depth.min(), rng))
            check(f"{cam['name']}: depth separates surfaces from empty space (some pixels hit, some don't)",
                  bool((depth < 0.99 * rng['far']).any()) and bool((depth > 0.99 * rng['far']).any()) or bool((depth < 0.99 * rng['far']).all()))
            check(f"{cam['name']}: depth only on request", 'depth' not in res)
            check(f"{cam['name']}: unknown encoding -> rejected", raises(Primitive('capture_image', camera=cam['name'], encoding='jpg'), world, 'rejected'))
        check('unknown camera -> rejected', raises(Primitive('capture_image', camera='nope'), world, 'rejected'))

        for r in robots:
            if world.act(Primitive('get_state', r['robot_id']))['gripper_opening'] is not None:
                test_gripper(world, r['robot_id'])

        print('get_state (frame)')
        for r in robots:
            sw = world.act(Primitive('get_state', r['robot_id']))
            sb = world.act(Primitive('get_state', r['robot_id'], frame='base'))
            check(f"{r['robot_id']}: default frame is reported as 'world'", sw['frame'] == 'world', sw['frame'])
            check(f"{r['robot_id']}: frame='base' is reported as 'base'", sb['frame'] == 'base', sb['frame'])
            check(f"{r['robot_id']}: robot at the origin -> same pose in both frames",
                  max(abs(a - b) for a, b in zip(sw['end_effector']['position'], sb['end_effector']['position'])) < 1e-6)
            check(f"{r['robot_id']}: unknown frame -> rejected", raises(Primitive('get_state', r['robot_id'], frame='tool'), world, 'rejected'))
        check("world: `frame` is ignored and reported as 'world'",
              world.act(Primitive('get_state', frame='base'))['frame'] == 'world')

        print('get_state (world)')
        w1 = world.act(Primitive('get_state'))
        time.sleep(0.3)
        w2 = world.act(Primitive('get_state'))
        check('simulation time advances', w2['time'] > w1['time'], (w1['time'], w2['time']))
        check('free bodies have position + orientation',
              all(len(b['position']) == 3 and len(b['orientation']) == 4 for b in w1['bodies'].values()))

        if robots:
            test_move_joints(world, args, robots[0])

        print('reset')
        test_reset(world, args, robots[0]['robot_id'] if robots else None, initial)

        print('error paths')
        check('unknown verb -> rejected', raises(Primitive('fly'), world, 'rejected'))
        check('unknown robot -> rejected', raises(Primitive('get_state', 'nobody'), world, 'rejected'))
        check('non-dict value -> rejected', _raw_value_rejected(world))

        print('robustness')
        check('server still answers after errors', world.act(Primitive('list_robots')) is not None)

    ok = all(results)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if ok else 1)


def test_move_joints(world, args, robot):
    rid, n = robot['robot_id'], robot['n_joints']
    home = world.act(Primitive('get_state', rid))['joint_positions']
    target = list(home)
    target[0] += 0.4

    print('move_joints (blocking)')
    t0 = time.time()
    res = world.act(Primitive('move_joints', rid, positions=target, speed=1.0))
    state = world.act(Primitive('get_state', rid))['joint_positions']
    check('reply only after arrival: error < tolerance', res['error'] < 0.01, res['error'])
    check('state matches the target afterwards', max(abs(a - b) for a, b in zip(state, target)) < 0.01)
    check('took real motion time (0.4 rad at 1 rad/s)', time.time() - t0 > 0.3, time.time() - t0)

    print('move_joints (one motion per robot)')
    slow = {}
    other = WorldController(args.host, args.port, timeout=30)
    thread = threading.Thread(target=lambda: slow.update(
        res=other.act(Primitive('move_joints', rid, positions=home, speed=0.2))))
    thread.start()
    time.sleep(0.5)
    check('second move on a busy robot -> rejected', raises(Primitive('move_joints', rid, positions=target), world, 'rejected'))
    mid = world.act(Primitive('get_state', rid))
    check('get_state still works mid-motion', 0.02 < abs(mid['joint_positions'][0] - target[0]) < 0.4 - 0.02)
    thread.join()
    other.close()
    check('first move completes at home', slow['res']['error'] < 0.01)

    print('move_joints (validation + failure)')
    check('wrong joint count -> rejected', raises(Primitive('move_joints', rid, positions=[0.0] * (n - 1)), world, 'rejected'))
    check('missing positions -> rejected', raises(Primitive('move_joints', rid), world, 'rejected'))
    check('world has no joints -> rejected', raises(Primitive('move_joints', positions=[0.0] * n), world, 'rejected'))
    check('target outside joint limits -> rejected', raises(Primitive('move_joints', rid, positions=[99.0] * n), world, 'rejected'))
    check('too short a timeout -> failed', raises(Primitive('move_joints', rid, positions=target, speed=0.1, timeout=0.2), world, 'failed'))
    world.act(Primitive('move_joints', rid, positions=home))
    check('robot not stuck busy after a failure', True)

    print('move_to_pose (inverse kinematics)')
    ee = world.act(Primitive('get_state', rid))['end_effector']
    goal = [ee['position'][0] + 0.05, ee['position'][1] - 0.05, ee['position'][2] - 0.05]
    res = world.act(Primitive('move_to_pose', rid, position=goal))
    now = world.act(Primitive('get_state', rid))['end_effector']
    check('arrives within 3 mm of the goal', sum((a - b) ** 2 for a, b in zip(now['position'], goal)) ** 0.5 < 0.003, res)
    dot = abs(sum(a * b for a, b in zip(now['orientation'], ee['orientation'])))
    check('orientation kept when none is given', dot > 0.999, dot)
    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))   # +20 deg about world z
    w, x, y, z = now['orientation']
    turned = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
    res = world.act(Primitive('move_to_pose', rid, position=goal, orientation=turned))
    check('explicit orientation reached', res['rotation_error'] < 0.02, res)
    check('unreachable pose -> failed', raises(Primitive('move_to_pose', rid, position=[5, 0, 0.5], timeout=2), world, 'failed'))
    check('bad position length -> rejected', raises(Primitive('move_to_pose', rid, position=[0, 0]), world, 'rejected'))
    check('unknown frame -> rejected', raises(Primitive('move_to_pose', rid, position=goal, frame='tool'), world, 'rejected'))
    res = world.act(Primitive('move_to_pose', rid, position=goal, frame='base'))   # robots sit at the origin: base == world
    check("frame='base' == 'world' for a robot at the origin", res['position_error'] < 0.003, res)
    check('zero quaternion -> rejected', raises(Primitive('move_to_pose', rid, position=goal, orientation=[0, 0, 0, 0]), world, 'rejected'))
    check('world has no end effector -> rejected', raises(Primitive('move_to_pose', position=goal), world, 'rejected'))
    world.act(Primitive('move_joints', rid, positions=home))


def test_reset(world, args, rid, initial):
    res = world.act(Primitive('reset'))
    check('reset returns the clock at ~0', res['time'] < 0.01, res)
    time.sleep(0.5)
    check('sim time restarted', world.act(Primitive('get_state'))['time'] < 2.0)
    time.sleep(3.0)   # the scene starts over from its initial poses (a cube dropped from 0.5 m needs a moment to land)
    now = world.act(Primitive('get_state'))
    moved = [n for n, b in initial['bodies'].items() if max(abs(a - c) for a, c in zip(b['position'], now['bodies'][n]['position'])) > 0.005]
    check('every body is back where it started, once settled (within 5 mm)', not moved, moved)
    if rid is None:
        return
    j = world.act(Primitive('get_state', rid))['joint_positions']
    check('robot joints are back at home (within 0.02 rad)', max(abs(a - b) for a, b in zip(j, initial['joints'][rid])) < 0.02, (j, initial['joints'][rid]))
    # a motion from another client is cancelled, and that client gets an error, not a hang
    other, got = WorldController(args.host, args.port, timeout=30), {}
    def slow():
        try:
            other.act(Primitive('move_joints', rid, positions=[q + 0.5 for q in j], speed=0.1))
            got['status'] = 'done'
        except ActionError as exc:
            got['status'], got['error'] = exc.reply['status'], exc.reply['error']
    thread = threading.Thread(target=slow)
    thread.start()
    time.sleep(0.5)
    world.act(Primitive('reset'))
    thread.join(10)
    other.close()
    check('motion in flight is cancelled: its client gets `failed`', got.get('status') == 'failed' and 'reset' in got.get('error', ''), got)
    time.sleep(0.3)
    j2 = world.act(Primitive('get_state', rid))['joint_positions']
    check('the cancelled motion left no trace (robot is at home)', max(abs(a - b) for a, b in zip(j2, initial['joints'][rid])) < 0.02, j2)
    check('robot is free to move again after a reset', world.act(Primitive('move_joints', rid, positions=[q + 0.1 for q in j2], speed=1.0))['error'] < 0.01)
    world.act(Primitive('reset'))


def test_gripper(world, rid):
    print(f'gripper ({rid})')
    verbs = {p['verb'] for p in world.act(Primitive('list_primitives'))['primitives']}
    check('verbs advertised: actuate_tool, grasp, release', {'actuate_tool', 'grasp', 'release'} <= verbs)
    res = world.act(Primitive('release', rid))
    check('release: fully open, not blocked', res['opening'] > 0.9 and not res['blocked'], res)
    check('get_state reports gripper_opening', abs(world.act(Primitive('get_state', rid))['gripper_opening'] - res['opening']) < 0.02)
    res = world.act(Primitive('actuate_tool', rid, state='close'))
    check('close on nothing: fully closed, not blocked', res['opening'] < 0.1 and not res['blocked'], res)
    res = world.act(Primitive('actuate_tool', rid, state=0.5))
    check('state 0.5: half open, reached (not blocked)', abs(res['opening'] - 0.5) < 0.06 and not res['blocked'], res)
    check('bad state -> rejected', raises(Primitive('actuate_tool', rid, state='wide'), world, 'rejected'))
    check('state out of range -> rejected', raises(Primitive('actuate_tool', rid, state=1.5), world, 'rejected'))
    check('world has no gripper -> rejected', raises(Primitive('grasp'), world, 'rejected'))
    world.act(Primitive('release', rid))

    cube = world.act(Primitive('get_state'))['bodies'].get('cube')
    if cube is None:
        return
    print(f'pick and lift the cube ({rid})')
    x, y, z = cube['position']
    world.act(Primitive('move_to_pose', rid, position=[x, y, z + 0.18]))
    world.act(Primitive('move_to_pose', rid, position=[x, y, z], speed=0.1))
    res = world.act(Primitive('grasp', rid))
    check('grasp closes on the cube: blocked, partly open', res['blocked'] and 0.1 < res['opening'] < 0.9, res)
    world.act(Primitive('move_to_pose', rid, position=[x, y, z + 0.2], speed=0.1))
    time.sleep(1.0)
    a = world.act(Primitive('get_state'))['bodies']['cube']['position']
    time.sleep(1.0)
    b = world.act(Primitive('get_state'))['bodies']['cube']['position']
    check('cube lifted 0.2 m with the hand (within 5 mm)', abs(a[2] - (z + 0.2)) < 0.005 and abs(a[0] - x) < 0.005, a)
    check('cube is held still (< 1 mm drift in 1 s)', abs(b[2] - a[2]) < 0.001, (a[2], b[2]))
    check('release drops it', world.act(Primitive('release', rid))['opening'] > 0.9)


def _raw_value_rejected(world):
    """Bypass Primitive to send a malformed envelope (value must be an object)."""
    import json
    world._ws.send(json.dumps({'request_id': 'x', 'robot_id': 'world', 'action': 'get_state', 'value': [1, 2]}))
    reply = json.loads(world._ws.recv(timeout=5))
    return reply['status'] == 'rejected'


if __name__ == '__main__':
    main()

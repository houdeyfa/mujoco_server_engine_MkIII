"""Geom groups vs. cameras, in-process (no server needed):   python tests/test_groups.py

A 2 cm thick red slab sits on the floor, 1 m below a camera looking straight down.
  group 0  normal object   -> in RGB, in depth (depth = 0.98 m, the slab's top)
  group 4  RGB_ONLY_GROUP  -> in RGB, NOT in depth (depth = 1.00 m, the floor)  <- ArUco stickers
  group 3  hidden          -> in neither
"""
import os
import sys

import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from client.world_controller import _decode_images
from primitives import RGB_ONLY_GROUP, Context, execute
from utils.engine_utils import build_model, camera_index


def capture(group):
    cfg = {'world': 'default',
           'cameras': [{'name': 'top', 'pos': [0, 0, 1.0], 'lookat': [0, 0, 0], 'fovy': 45, 'resolution': [320, 240]}],
           'objects': [{'name': 'slab', 'type': 'box', 'size': [0.1, 0.1, 0.01], 'pos': [0, 0, 0.01], 'free': False,
                        'rgba': [1, 0, 0, 1], 'group': group, 'collide': False}]}
    m = build_model(cfg)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ctx = Context(m, d, cfg, {}, cameras=camera_index(m, cfg))
    rep = execute(ctx, {'robot_id': 'world', 'action': 'capture_image', 'value': {'camera': 'top', 'depth': True}})
    assert rep['status'] == 'done', rep
    r = _decode_images(rep['result'])
    rgb = r['rgb'][120, 160]
    return bool(rgb[0] > 200 and rgb[1] < 160 and rgb[2] < 160), float(r['depth'][120, 160])   # red-ish, not floor-white


results = []
for label, group, want_red, want_depth in [('group 0', 0, True, 0.98), ('RGB-only group', RGB_ONLY_GROUP, True, 1.00), ('group 3', 3, False, 1.00)]:
    red, depth = capture(group)
    ok = red == want_red and abs(depth - want_depth) < 1e-3
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: red in RGB = {red} (want {want_red}), depth = {depth:.3f} (want {want_depth})")
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

"""ArUco markers, in-process (no server):   python tests/test_markers.py
A camera 0.5 m above the floor looks straight down; OpenCV decodes what it sees."""
import math
import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from client.world_controller import _decode_images
from primitives import Context, execute
from utils.engine_utils import build_model, camera_index

DET = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
results = []


def check(name, ok, detail=''):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ''))


def look(markers, objects=(), move=None):
    cfg = {'world': 'default', 'markers': markers, 'objects': list(objects),
           'cameras': [{'name': 'top', 'pos': [0, 0, 0.5], 'lookat': [0, 0, 0], 'fovy': 45, 'resolution': [640, 480]}]}
    m = build_model(cfg)
    d = mujoco.MjData(m)
    if move:
        move(m, d)
    mujoco.mj_forward(m, d)
    ctx = Context(m, d, cfg, {}, cameras=camera_index(m, cfg))
    r = _decode_images(execute(ctx, {'robot_id': 'world', 'action': 'capture_image', 'value': {'camera': 'top', 'depth': True}})['result'])
    corners, ids, _ = DET.detectMarkers(cv2.cvtColor(r['rgb'], cv2.COLOR_RGB2GRAY))
    found = {} if ids is None else {int(i): c[0] for i, c in zip(ids.ravel(), corners)}
    return found, r['depth']


def signs(c):   # corner order as seen in the image, relative to the marker centre (x right, y down)
    return np.sign(c - c.mean(0)).astype(int).tolist()


tag = dict(name='a', id=20, size=0.10, margin=0.02, position=[0, 0, 0.001])
found, depth = look([{**tag, 'orientation': [0, 0, 0]}])
check('id 20 decodes (un-mirrored)', 20 in found, list(found))
check('orientation [0,0,0]: corners TL,TR,BR,BL (+X right, +Y up)', 20 in found and signs(found[20]) == [[-1, -1], [1, -1], [1, 1], [-1, 1]])
size_px = np.ptp(found[20][:, 0]) if 20 in found else 0
check('printed size: 0.10 m at 0.5 m is ~ 0.10 * f / 0.5 px', abs(size_px - 0.10 * (240 / math.tan(math.radians(22.5))) / 0.5) < 6, size_px)
check('depth ignores the sticker (floor stays at 0.500 m)', abs(float(depth[240, 320]) - 0.5) < 1e-3, depth[240, 320])

found, _ = look([{**tag, 'orientation': [0, 0, math.pi / 2]}])
check('yaw +90 deg: marker turns counter-clockwise seen from above (corner 0 TL -> BL)', 20 in found and signs(found[20]) == [[-1, 1], [-1, -1], [1, -1], [1, 1]], found.get(20) is not None and signs(found[20]))

found, _ = look([{**tag, 'id': 7, 'position': [0.1, 0.05, 0.001]}, {**tag, 'name': 'b', 'id': 21, 'position': [-0.1, -0.05, 0.001]}])
check('two markers, two ids', set(found) == {7, 21}, list(found))

cube = dict(name='cube', type='box', size=[0.05] * 3, pos=[0, 0, 0.05])
on_cube = {**tag, 'attach_to': {'body': 'cube'}, 'position': [0, 0, 0.0505]}
before, _ = look([on_cube], [cube])
after, _ = look([on_cube], [cube], move=lambda m, d: d.qpos.__setitem__(slice(0, 3), [0.08, 0, 0.05]))
check('attach_to: marker rides the cube', 20 in before and 20 in after and after[20][:, 0].mean() - before[20][:, 0].mean() > 80,
      (before.get(20) is not None and before[20][:, 0].mean(), after.get(20) is not None and after[20][:, 0].mean()))

for label, bad in [('id outside the dictionary', {**tag, 'id': 999}), ('unknown dictionary', {**tag, 'dictionary': 'DICT_NOPE'}),
                   ('attach_to a missing body', {**tag, 'attach_to': {'body': 'ghost'}})]:
    try:
        look([bad])
        check(f'{label} -> ValueError', False)
    except ValueError:
        check(f'{label} -> ValueError', True)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

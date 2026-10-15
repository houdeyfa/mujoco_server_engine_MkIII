"""Pick the red cube and put it somewhere else. Only primitives, no server imports.

    mjpython server.py config/ur10e_robotiq.yaml --gui      (terminal 1)
    python client/examples/pick_place.py ur                        (terminal 2)      # or: panda, with config/panda.yaml
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from world_controller import Primitive, WorldController

ROBOT = sys.argv[1] if len(sys.argv) > 1 else 'ur'
PLACE_XY = {'ur': [0.35, 0.0], 'panda': [0.45, 0.20]}[ROBOT]   # hardcoded: where the cube should end up (world frame)
HOVER = 0.18      # clearance above the cube while travelling, metres


def move(w, x, y, z, speed=0.25):
    return w.act(Primitive('move_to_pose', ROBOT, position=[x, y, z], speed=speed))   # orientation kept: gripper stays pointing down


with WorldController() as w:
    start = w.act(Primitive('get_state', ROBOT))['joint_positions']
    x, y, z = w.act(Primitive('get_state'))['bodies']['cube']['position']
    px, py = PLACE_XY
    print(f'cube at ({x:.3f}, {y:.3f}, {z:.3f}) -> placing it at ({px:.3f}, {py:.3f})')

    print('open the hand');                 w.act(Primitive('release', ROBOT))
    print('hover above the cube');          move(w, x, y, z + HOVER)
    print('go down to it');                 move(w, x, y, z, speed=0.1)
    print('grasp');                         held = w.act(Primitive('grasp', ROBOT))
    if not held['blocked']:
        sys.exit('the fingers closed on nothing: the cube is not between them')
    print(f"  holding it (hand {held['opening']:.0%} open)")
    print('lift');                          move(w, x, y, z + HOVER, speed=0.1)
    print('carry it over the target');      move(w, px, py, z + HOVER)
    print('lower it');                      move(w, px, py, z + 0.003, speed=0.1)   # 3 mm above the floor
    print('let go');                       w.act(Primitive('release', ROBOT))
    print('back up');                       move(w, px, py, z + HOVER, speed=0.1)
    print('home');                         w.act(Primitive('move_joints', ROBOT, positions=start))

    fx, fy, fz = w.act(Primitive('get_state'))['bodies']['cube']['position']
    err = ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5
    print(f'cube ended at ({fx:.3f}, {fy:.3f}, {fz:.3f}): {err * 1000:.1f} mm from the target')
    sys.exit(0 if err < 0.01 else 'missed the target by more than 1 cm')

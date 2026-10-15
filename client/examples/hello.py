"""python server.py config/ur10e_robotiq.yaml   (terminal 1)   then   python client/examples/hello.py   (terminal 2)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from world_controller import ActionError, Primitive, WorldController

with WorldController() as world:
    for p in world.act(Primitive('list_primitives'))['primitives']:
        print(' ', p['verb'], '-', p['description'], p['args'])
    robots = world.act(Primitive('list_robots'))['robots']
    print('robots:', robots)
    for r in robots:
        s = world.act(Primitive('get_state', r['robot_id']))
        print(r['robot_id'], 'joints', [round(q, 3) for q in s['joint_positions']], 'ee', s['end_effector'])
    print('world:', world.act(Primitive('get_state'))['bodies'])
    try:
        world.act(Primitive('fly', 'ur'))
    except ActionError as exc:
        print('expected error ->', exc)

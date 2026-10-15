# MuJoCo server

A MuJoCo world that clients drive over a WebSocket with primitive commands (`move_to_pose`, `grasp`, `capture_image`, ...).
The scene (robots, objects, cameras, markers) comes from a YAML file.

## 1. Get the code

```bash
git clone https://github.com/houdeyfa/mujoco_server_engine_MkIII.git mujoco_server
cd mujoco_server
pip install mujoco websockets pyyaml numpy pillow opencv-python coacd trimesh
```

`coacd` and `trimesh` are only needed for gear/mesh objects, `opencv-python` for ArUco markers.

## 2. Get the robot models (MuJoCo Menagerie)

The robots (UR10e, Panda, grippers) are not in this repo. Clone Menagerie next to your home folder:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie
```

The server looks for it at `~/mujoco_menagerie`. Elsewhere? Either set `export MUJOCO_MENAGERIE=/path/to/mujoco_menagerie`
or put `menagerie: /path/to/mujoco_menagerie` in a config file.

## 3. Run it

```bash
python server.py config/ur10e_robotiq.yaml               # no window
mjpython server.py config/ur10e_robotiq.yaml --gui       # with the viewer (macOS needs mjpython)
python client/examples/pick_place.py ur                  # in a second terminal: a client picks and places the cube
```

Only one server at a time: it listens on port 8988 (`--port N` to change it).
Tests: start a server, then `python tests/fake_running_client.py`.

## World files (`worlds/`)

A world is the environment: physics, floor, sky, lights. Pick one in a config with `world: checker`
(a name in `worlds/`, or a path). No `world:` key means `worlds/default.yaml`.

```yaml
physics:
  timestep: 0.002
  gravity: [0, 0, -9.81]
  contact_override:          # one stiff contact setting for everything; delete the block for MuJoCo's soft default
    solref: [0.004, 1]       # solref[0] must be at least 2 x timestep
    solimp: [0.99, 0.99, 0.001, 0.5, 2]
floor:
  texture: checker           # or none, then `color: [r, g, b, a]`
  colors:
    - [0.2, 0.3, 0.4]
    - [0.3, 0.4, 0.5]
  square: 0.1                # metres
sky:
  top: [0.4, 0.6, 0.8]
  bottom: [0, 0, 0]
lights:
  - pos: [0, 0, 3]
    dir: [0, 0, -1]
    diffuse: 0.8
```

Copy a world, change what you want, and point a config at it. A `simulation:` block in a config overrides the world's `physics:`.

## Config files (`config/`)

A config is one scene: what is in it, not how the world behaves.

```yaml
world: checker               # worlds/checker.yaml
server: {port: 8988}

robots:
  - name: ur                 # the id clients use: Primitive('move_to_pose', 'ur', ...)
    model: universal_robots_ur10e/ur10e.xml     # path inside Menagerie
    pos: [0, 0, 0]
    euler: [0, 0, 0]         # degrees
    gravcomp: true           # the arm floats (no sag)
    home: [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]   # start joints in radians, or the name of a keyframe
    tool:                    # a gripper bolted on the flange (a second Menagerie file)
      model: robotiq_2f85/2f85.xml
      body: base_mount
      site: attachment_site
    gripper:                 # the hand's actuator: strength, and which ctrl end is "open"
      actuator: tool/fingers_actuator
      kp: 100
      force: 5

cameras:
  - name: overview           # fixed in the air; position and the point it looks at
    pos: [1.6, -1.6, 1.4]
    lookat: [0.3, 0.0, 0.3]
    fovy: 45
    resolution: [640, 480]

objects:
  - name: cube
    type: box                # box | sphere | cylinder | capsule
    size: [0.03, 0.03, 0.03] # half sizes, metres
    pos: [0.6, 0.2, 0.03]
    rgba: [0.9, 0.2, 0.2, 1]
    mass: 0.1
  - name: sun                # a mesh instead of a primitive
    mesh: assets/gearbox/gear.stl
    scale: 0.004             # metres per STL unit
    collision: parts         # slice it into convex pieces (cached in .mesh_cache/) so teeth and holes collide

markers:                     # ArUco stickers: visible in the image, not in depth
  - name: ref_a
    id: 20
    dictionary: DICT_4X4_50
    size: 0.05               # black square, metres
    margin: 0.01
    position: [0.35, 0.40, 0.001]
    orientation: [0, 0, 0]   # roll, pitch, yaw in radians
```

Every block is optional. Look at `config/ur10e_robotiq.yaml`, `config/panda.yaml` and `config/ur10e_gearbox.yaml` for full examples.
Print the markers of a config: `python aruco_config/markers.py config/ur10e_robotiq.yaml out/`.

## Credits

This project uses the robot models of [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Zakka, Tassa and the Menagerie contributors), cited as:

```bibtex
@software{menagerie2022github,
  author = {Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}},
  title = {{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}},
  url = {http://github.com/google-deepmind/mujoco_menagerie},
  year = {2022},
}
```

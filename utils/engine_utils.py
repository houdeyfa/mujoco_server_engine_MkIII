"""World building from a scene config: no networking, no loop, no viewer.

Everything here turns a parsed YAML dict into a compiled MjModel (plus its
initial MjData state) and can be imported and tested on its own.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from aruco_config.markers import add_markers
from utils.camera_utils import intrinsics, look_at_quat
from utils.mesh_utils import convex_parts

GEOMS = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


def menagerie_path(cfg: dict, rel: str) -> str:
    """`rel` is relative to the Menagerie checkout unless it is already a real path."""
    if os.path.isabs(rel) or os.path.exists(rel):
        return rel
    root = cfg.get("menagerie") or os.environ.get("MUJOCO_MENAGERIE") or "~/mujoco_menagerie"
    return os.path.join(os.path.expanduser(root), rel)


def tune_position_actuator(spec: mujoco.MjSpec, name: str | None, kp: float | None, force: float | None) -> None:
    """Re-gain a position servo the way speedrun12 does for the 2F-85 (Menagerie's gains are weak on some
    robots): force = gain * ctrl - kp * q - kv * qdot, with kv = kp / 10 and a clipped force.
    The ctrl -> target scale (0.8 / 255 rad per ctrl unit on the 2F-85) is read from the actuator itself."""
    a = spec.actuator(name) if name else spec.actuators[0]
    if a.biasprm[1] == 0:
        raise ValueError(f"actuator {a.name!r} is not a position servo (biasprm[1] == 0); cannot set kp")
    if kp is not None:
        scale = a.gainprm[0] / -a.biasprm[1]
        a.gainprm[0] = kp * scale
        a.biasprm[1] = -kp
        a.biasprm[2] = -kp / 10
    if force is not None:
        a.forcerange = [-force, force]


def add_robot(scene: mujoco.MjSpec, cfg: dict, r: dict) -> None:
    spec = mujoco.MjSpec.from_file(menagerie_path(cfg, r["model"]))

    for site in r.get("sites", []):   # extra named sites, e.g. a `tcp` between the fingertips
        spec.body(site["body"]).add_site(name=site["name"], pos=site.get("pos", [0, 0, 0]))

    tool = r.get("tool")
    if tool:  # bolt the tool onto the robot's flange site, both files untouched
        t = mujoco.MjSpec.from_file(menagerie_path(cfg, tool["model"]))
        t_body = t.body(tool["body"]) if "body" in tool else t.worldbody.first_body()
        spec.site(tool.get("site", "attachment_site")).attach_body(t_body, "tool/", "")

    g = r.get("gripper")   # grip strength; `actuator` is named as in the robot file, tool ones as `tool/<name>`
    if g:
        tune_position_actuator(spec, g.get("actuator"), g.get("kp"), g.get("force"))

    if r.get("gravcomp"):  # robot floats, no sag: handy while there is no controller
        for b in spec.bodies:
            b.gravcomp = 1

    base = spec.worldbody.first_body()
    frame = scene.worldbody.add_frame(pos=r.get("pos", [0, 0, 0]), euler=r.get("euler", [0, 0, 0]))
    frame.attach_body(base, f"{r['name']}/", "")


def add_mesh_geoms(scene: mujoco.MjSpec, body, o: dict, kw: dict) -> None:
    """A mesh object. `collision: parts` (the speedrun's way) draws the real mesh with no collision and collides
    with its convex pieces, hidden in group 3 and massless: use it for gears, anything with holes or teeth.
    Without it the whole mesh collides as one convex hull. Meshes are loaded once per (file, scale), so four
    planets share one gear. `scale` = metres per STL unit."""
    s = o.get("scale", 1.0)

    def mesh(path, **extra):   # the full path in the name: piece "0.stl" of the gear and of the ring must not clash
        name = f"{path}@{s}" + ("+exact" if extra else "")
        if all(m.name != name for m in scene.meshes):
            scene.add_mesh(name=name, file=path, scale=[s, s, s], **extra)
        return name

    if o.get("collision") != "parts":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh(o["mesh"]), **kw)
        return
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh(o["mesh"], inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_EXACT),
                  **{**kw, "contype": 0, "conaffinity": 0})   # drawn, mass from its volume, never collides
    for piece in convex_parts(o["mesh"], o.get("threshold", 0.015)):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh(piece), density=0, group=3,
                      rgba=[0, 1, 1, 0.5], friction=kw["friction"])


def add_object(scene: mujoco.MjSpec, o: dict) -> None:
    body = scene.worldbody.add_body(name=o["name"], pos=o.get("pos", [0, 0, 0]),
                                    euler=o.get("euler", [0, 0, 0]))
    if o.get("free", True):
        body.add_freejoint()
    kw = dict(rgba=o.get("rgba", [0.7, 0.7, 0.7, 1]), friction=o.get("friction", [1, 0.005, 0.0001]),
              group=o.get("group", 0))      # see primitives.RGB_ONLY_GROUP: 4 = drawn in RGB, invisible to depth
    if not o.get("collide", True):          # a ghost: drawn, but nothing touches it (stickers, markers)
        kw["contype"] = kw["conaffinity"] = 0
    if "mass" in o:
        kw["mass"] = o["mass"]
    if "mesh" in o:
        add_mesh_geoms(scene, body, o, kw)
    else:
        body.add_geom(type=GEOMS[o["type"]], size=o["size"], **kw)


INTEGRATORS = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
               "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "rk4": mujoco.mjtIntegrator.mjINT_RK4}
CONES = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}
WORLDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worlds")


def load_world(cfg: dict) -> dict:
    """The environment (physics, floor, sky, lights). `world:` in the scene config names a YAML file,
    as a path or a name inside worlds/; without it worlds/default.yaml is used. The config's own
    `simulation:` block overrides the world's `physics:` for one-off tweaks."""
    import yaml
    ref = cfg.get("world", "default")
    candidates = [ref, os.path.join(WORLDS_DIR, ref), os.path.join(WORLDS_DIR, ref + ".yaml")]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        raise FileNotFoundError(f"world {ref!r} not found; tried {candidates}")
    with open(path) as fh:
        world = yaml.safe_load(fh) or {}
    world["physics"] = {**world.get("physics", {}), **cfg.get("simulation", {})}
    return world


def add_world(scene: mujoco.MjSpec, world: dict) -> None:
    ph = world["physics"]
    scene.option.timestep = ph.get("timestep", 0.002)
    scene.option.integrator = INTEGRATORS[ph.get("integrator", "implicitfast")]
    scene.option.cone = CONES[ph.get("cone", "elliptic")]
    scene.option.impratio = ph.get("impratio", 10)
    scene.option.gravity = ph.get("gravity", [0, 0, -9.81])
    scene.memory = ph.get("memory", 200 * 1024 * 1024)

    # One solref/solimp for EVERY contact (fingers, objects, floor): ignores what each geom says, so
    # Menagerie stays untouched. Set on OUR spec: when attaching, the parent's <option> wins.
    ov = ph.get("contact_override")
    if ov:
        solref, solimp = ov.get("solref", [0.004, 1]), ov.get("solimp", [0.99, 0.99, 0.001, 0.5, 2])
        if solref[0] < 2 * scene.option.timestep:   # MuJoCo would silently raise it: fail loudly instead
            raise ValueError(f"contact_override.solref[0]={solref[0]} < 2 x timestep ({2 * scene.option.timestep}); "
                             f"lower the timestep or raise solref[0]")
        scene.option.o_solref = solref
        scene.option.o_solimp = solimp
        scene.option.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE

    sky = world.get("sky")
    if sky:
        scene.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                          rgb1=sky.get("top", [0.4, 0.6, 0.8]), rgb2=sky.get("bottom", [0, 0, 0]),
                          width=512, height=3072)

    floor = world.get("floor", {})
    geom = dict(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], name="floor")
    if floor.get("texture", "none") == "checker":
        c1, c2 = floor.get("colors", [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])
        scene.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=c1, rgb2=c2,
                          mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8], width=300, height=300)
        mat = scene.add_material(name="floor", texuniform=True, reflectance=floor.get("reflectance", 0.0))
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
        repeat = 1.0 / floor.get("square", 0.1)        # measured: repeats per metre = 1 / square side
        mat.texrepeat = [repeat, repeat]
        geom["material"] = "floor"
    else:
        geom["rgba"] = floor.get("color", [0.85, 0.85, 0.85, 1])
    scene.worldbody.add_geom(**geom)

    for light in world.get("lights", []):
        d = light.get("diffuse", 0.8)
        scene.worldbody.add_light(pos=light.get("pos", [0, 0, 3]), dir=light.get("dir", [0, 0, -1]),
                                  diffuse=[d] * 3 if isinstance(d, (int, float)) else d)


def add_cameras(scene: mujoco.MjSpec, cfg: dict) -> None:
    """Fixed cameras from the scene config's `cameras:` list, floating in the world."""
    seen, widest, tallest = set(), 640, 480        # MuJoCo's default offscreen buffer is 640 x 480
    for c in cfg.get("cameras", []):
        name = c["name"]
        if name in seen:
            raise ValueError(f"duplicate camera name {name!r}")
        seen.add(name)
        width, height = c.get("resolution", [640, 480])
        widest, tallest = max(widest, width), max(tallest, height)
        scene.worldbody.add_camera(name=name, pos=c["pos"], quat=look_at_quat(c["pos"], c["lookat"]),
                                   fovy=c.get("fovy", 45.0), resolution=[width, height])
    scene.visual.global_.offwidth, scene.visual.global_.offheight = widest, tallest   # renderer needs room


def camera_index(model: mujoco.MjModel, cfg: dict) -> dict:
    """camera name -> what list_cameras / capture_image need. No pose: see camera_utils."""
    out = {}
    for c in cfg.get("cameras", []):
        width, height = c.get("resolution", [640, 480])
        fovy = float(model.cam_fovy[model.camera(c["name"]).id])
        out[c["name"]] = {"width": width, "height": height, "fovy": fovy, "intrinsics": intrinsics(width, height, fovy)}
    return out


def build_model(cfg: dict) -> mujoco.MjModel:
    scene = mujoco.MjSpec()
    add_world(scene, load_world(cfg))
    for r in cfg.get("robots", []):
        add_robot(scene, cfg, r)
    for o in cfg.get("objects", []):
        add_object(scene, o)
    add_markers(scene, cfg)     # after the objects: a marker can ride on one (attach_to)
    add_cameras(scene, cfg)
    return scene.compile()


def apply_home(model: mujoco.MjModel, data: mujoco.MjData, cfg: dict) -> None:
    """`home:` is either a list (sets the first N arm joints and holds them with ctrl) or the NAME of a
    keyframe from the robot's own file (e.g. `home` on the Panda). A keyframe is copied for THIS robot's
    joints and actuators only, never `mj_resetDataKeyframe`, which would reset every robot and object."""
    J, A = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
    for r in cfg.get("robots", []):
        home = r.get("home")
        if not home:
            continue
        prefix = f"{r['name']}/"
        joints = [j for j in range(model.njnt) if (mujoco.mj_id2name(model, J, j) or "").startswith(prefix)]
        acts = [a for a in range(model.nu) if (mujoco.mj_id2name(model, A, a) or "").startswith(prefix)]
        if isinstance(home, str):
            key = model.key(prefix + home)
            for j in joints:
                data.qpos[model.jnt_qposadr[j]] = key.qpos[model.jnt_qposadr[j]]
            for a in acts:
                data.ctrl[a] = key.ctrl[a]
        else:
            for k, q in enumerate(home):
                data.qpos[model.jnt_qposadr[joints[k]]] = q
                data.ctrl[acts[k]] = q
    mujoco.mj_forward(model, data)


def _name(model, kind, i):
    return mujoco.mj_id2name(model, kind, i) or ""


def robot_index(model: mujoco.MjModel, cfg: dict) -> dict:
    """robot_id -> ids the verbs need. Arm joints are `<name>/x`, tool joints `<name>/tool/x`."""
    index = {}
    for r in cfg.get("robots", []):
        prefix = f"{r['name']}/"
        joints = [j for j in range(model.njnt) if _name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(prefix)]
        non_tool = [j for j in joints if not _name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(prefix + "tool/")]
        # joints with their own actuator are the arm; a hand baked into the robot file (Panda fingers)
        # is driven through a tendon, so it lands in tool_joints. No actuators at all: everything is arm.
        driven = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)
                  if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        arm = [j for j in non_tool if j in driven] or non_tool
        # end effector: explicit `end_effector:` (site or body, relative to the robot), else the tool's pinch, else the flange
        ee = None
        for rel in [r.get("end_effector"), "tool/pinch", "attachment_site"]:
            for kind in (mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_BODY):
                if rel and mujoco.mj_name2id(model, kind, prefix + rel) >= 0:
                    ee = (kind, mujoco.mj_name2id(model, kind, prefix + rel))
                    break
            if ee:
                break
        # the robot's frame = the pose it was PLACED with (`pos` / `euler` in the config). Not the base body:
        # Menagerie bodies carry their own rotation (the UR10e's base is turned 180 deg about z).
        frame_quat = np.zeros(4)
        mujoco.mju_euler2Quat(frame_quat, np.radians(r.get("euler", [0, 0, 0])), "xyz")
        # gripper: the actuator that opens/closes the hand, and which end of its ctrlrange is "open"
        gripper, g = None, r.get("gripper")
        if g and "actuator" in g:
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + g["actuator"])
            lo, hi = model.actuator_ctrlrange[act]
            open_ctrl = g.get("open_ctrl", lo)
            gripper = {"act": act, "open": float(open_ctrl), "closed": float(hi if open_ctrl == lo else lo),
                       "scale": float(model.actuator_gainprm[act, 0] / -model.actuator_biasprm[act, 1])}
        index[r["name"]] = {"gripper": gripper, "frame_pos": np.array(r.get("pos", [0, 0, 0]), dtype=float), "frame_quat": frame_quat, "arm_joints": arm, "arm_actuators": [driven.get(j) for j in arm], "tool_joints": [j for j in joints if j not in arm],
                            "ee": ee, "ik": r.get("ik", {}), "tool": r.get("tool", {}).get("model")}
    return index

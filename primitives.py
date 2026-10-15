"""Verbs the server understands, and the request/reply envelope around them.

    request  {"request_id": str, "robot_id": "world" | "<robot>", "action": "<verb>", "value": {args}}
    reply    {"request_id": str, "status": "done" | "failed" | "rejected", "result": {...} | None, "error": str | None}

Add a verb by decorating a function with @verb("name"); the client never changes.
Handlers run on the physics thread between two steps, so they may read and write
MuJoCo state freely. Conventions: SI units, quaternions are MuJoCo's (w, x, y, z).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from utils.camera_utils import RGB_ONLY_GROUP, encode_image
from utils.engine_utils import apply_home
from utils.robot_utils import Settle, ee_pose, ik_step, pose_error, pose_from_world, pose_to_world, step_target

WORLD = "world"
FRAMES = ("world", "base")   # base = the pose the robot was placed with (pos/euler in the config)
VERBS: dict[str, Callable] = {}
ARGS: dict[str, dict] = {}      # verb -> {arg name: type/description}, advertised by list_primitives


class Rejected(Exception):
    """The request itself is wrong (unknown verb/robot, bad args). Nothing was run."""


@dataclass
class Context:
    model: mujoco.MjModel
    data: mujoco.MjData
    cfg: dict
    robots: dict            # robot_id -> ids, see engine_utils.robot_index
    active: dict = field(default_factory=dict)   # robot_id -> running Task (one motion per robot)
    cameras: dict = field(default_factory=dict)  # name -> {width, height, fovy, intrinsics}, see engine_utils.camera_index
    renderers: dict = field(default_factory=dict)   # (width, height) -> mujoco.Renderer, created on first capture


def verb(name: str, args: dict | None = None):
    """Register a verb. `args` documents its named args; the docstring's first line is its description."""
    def register(fn):
        VERBS[name] = fn
        ARGS[name] = args or {}
        return fn
    return register


def _reply(req, status="done", result=None, error=None):
    return {"request_id": req.get("request_id"), "status": status, "result": result, "error": error}


def _failure(req, exc):
    if isinstance(exc, Rejected):
        return _reply(req, "rejected", error=str(exc))
    return _reply(req, "failed", error=f"{type(exc).__name__}: {exc}")


class Task:
    """A blocking verb in flight. Its handler is a generator: every `yield` means "one more physics
    step please"; `return value` is the result; raising fails the request. The reply is only sent
    when the generator finishes."""

    def __init__(self, ctx, robot_id, req, gen):
        self.ctx, self.robot_id, self.req, self.gen = ctx, robot_id, req, gen
        self.cancelled = False

    def step(self):
        """Advance to the next yield. Returns the reply once finished, else None."""
        try:
            if self.cancelled:           # `reset` stopped it: never touch the new world, just answer its client
                self.gen.close()
                raise RuntimeError("cancelled by reset")
            next(self.gen)
            return None
        except StopIteration as stop:
            reply = _reply(self.req, result=stop.value)
        except Exception as exc:
            reply = _failure(self.req, exc)
        if self.ctx.active.get(self.robot_id) is self:
            del self.ctx.active[self.robot_id]
        return reply


def execute(ctx: Context, req: dict):
    """Run one request. Returns a reply dict, or a Task if the verb blocks (caller steps it each tick).
    Never raises."""
    try:
        robot_id, action, args = req.get("robot_id", WORLD), req.get("action"), req.get("value") or {}
        if not isinstance(args, dict):
            raise Rejected("'value' must be an object of named args")
        if action not in VERBS:
            raise Rejected(f"unknown action {action!r}; known: {sorted(VERBS)}")
        if robot_id != WORLD and robot_id not in ctx.robots:
            raise Rejected(f"unknown robot_id {robot_id!r}; known: {sorted(ctx.robots)}")
        if not inspect.isgeneratorfunction(VERBS[action]):
            return _reply(req, result=VERBS[action](ctx, robot_id, args))
        if robot_id in ctx.active:
            raise Rejected(f"{robot_id} is busy with {ctx.active[robot_id].req.get('action')!r}")
        task = Task(ctx, robot_id, req, VERBS[action](ctx, robot_id, args))
        ctx.active[robot_id] = task
        reply = task.step()          # runs the handler's validation up to its first yield
        return task if reply is None else reply
    except Exception as exc:
        return _failure(req, exc)


# ---- verbs ------------------------------------------------------------------

@verb("reset")
def reset(ctx, robot_id, args):
    """Put the whole world back exactly as it was when the server started: every body, joint, command and the
    clock. Any motion still in progress is cancelled (its client gets a `failed` reply). Objects start settling
    again from their initial poses, so give the scene a moment before reading it."""
    for task in ctx.active.values():
        task.cancelled = True
    mujoco.mj_resetData(ctx.model, ctx.data)
    apply_home(ctx.model, ctx.data, ctx.cfg)
    return {"time": ctx.data.time}


@verb("list_primitives")
def list_primitives(ctx, robot_id, args):
    """Every verb the server understands, with its description and named args."""
    return {"primitives": [{"verb": v, "description": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
                            "args": ARGS[v]} for v, fn in sorted(VERBS.items())]}


@verb("list_robots")
def list_robots(ctx, robot_id, args):
    """Robots in the scene: id, tool model, number of arm joints."""
    return {"robots": [{"robot_id": rid, "tool": r["tool"], "n_joints": len(r["arm_joints"])}
                       for rid, r in ctx.robots.items()]}


def _gripper(ctx, robot_id):
    if robot_id == WORLD or ctx.robots[robot_id]["gripper"] is None:
        raise Rejected(f"{robot_id!r} has no gripper; give its config a `gripper:` block with an `actuator`")
    return ctx.robots[robot_id]["gripper"]


def _opening(ctx, g) -> float:
    """Measured opening of the hand, 1.0 = fully open, 0.0 = fully closed. The servo's tendon length / its
    ctrl scale is the position in ctrl units, so one formula works for every hand."""
    ctrl_equiv = ctx.data.actuator_length[g["act"]] / g["scale"]
    return float(np.clip(1.0 - (ctrl_equiv - g["open"]) / (g["closed"] - g["open"]), 0.0, 1.0))


def _drive_gripper(ctx, robot_id, target, speed, timeout):
    """Ramp the hand's command to `target` opening (0..1) at `speed` strokes/s, wait until the fingers stop.
    `blocked` = they stopped short of the target: something is between the fingers (a grasp)."""
    g = _gripper(ctx, robot_id)
    if not 0.0 <= target <= 1.0 or speed <= 0:
        raise Rejected("opening must be in [0, 1] and 'speed' > 0")
    m, d = ctx.model, ctx.data
    goal_ctrl = g["open"] + (g["closed"] - g["open"]) * (1.0 - target)
    step = abs(g["closed"] - g["open"]) * speed * m.opt.timestep   # ramp: a jump would snap the fingers shut
    joints = ctx.robots[robot_id]["tool_joints"]
    qadr = [m.jnt_qposadr[j] for j in joints]
    settle, t_end = Settle(m.opt.timestep), d.time + 1.0 / speed + timeout
    while True:
        d.ctrl[g["act"]] += np.clip(goal_ctrl - d.ctrl[g["act"]], -step, step)
        yield
        still = settle.update(d.qpos[qadr])
        if still and abs(d.ctrl[g["act"]] - goal_ctrl) < 1e-9:
            opening = _opening(ctx, g)
            return {"opening": opening, "blocked": abs(opening - target) > 0.05, "time": d.time}
        if d.time > t_end:
            raise TimeoutError(f"gripper did not settle (opening {_opening(ctx, g):.2f}, target {target:.2f})")


_GRIP_ARGS = {"speed": "strokes per second the command ramps at (default 2.0)",
              "timeout": "extra sim seconds to settle after the ramp (default 3.0)"}


@verb("actuate_tool", args={"state": "'open', 'close' or a number 0..1 (1 = fully open, 0 = fully closed)", **_GRIP_ARGS})
def actuate_tool(ctx, robot_id, args):
    """Move the gripper to an opening and wait until the fingers stop. `blocked` says they stopped short of it."""
    state = args.get("state")
    target = {"open": 1.0, "close": 0.0}.get(state, state) if isinstance(state, (str, int, float)) else None
    if isinstance(target, str) or target is None or isinstance(target, bool):
        raise Rejected("'state' must be 'open', 'close' or a number 0..1")
    return (yield from _drive_gripper(ctx, robot_id, float(target), float(args.get("speed", 2.0)), float(args.get("timeout", 3.0))))


@verb("grasp", args=_GRIP_ARGS)
def grasp(ctx, robot_id, args):
    """Close the gripper until the fingers stop. `blocked: true` means they closed on something (holding it)."""
    return (yield from _drive_gripper(ctx, robot_id, 0.0, float(args.get("speed", 2.0)), float(args.get("timeout", 3.0))))


@verb("release", args=_GRIP_ARGS)
def release(ctx, robot_id, args):
    """Open the gripper fully and wait until the fingers stop."""
    return (yield from _drive_gripper(ctx, robot_id, 1.0, float(args.get("speed", 2.0)), float(args.get("timeout", 3.0))))


@verb("list_cameras")
def list_cameras(ctx, robot_id, args):
    """Cameras in the scene: name, resolution, field of view and intrinsics (never their pose)."""
    return {"cameras": [{"name": n, **{k: c[k] for k in ("width", "height", "fovy", "intrinsics")}}
                        for n, c in ctx.cameras.items()]}


@verb("capture_image", args={"camera": "camera name, see list_cameras",
                             "encoding": "'png' (default, lossless) or 'raw' (uint8 RGB bytes); applies to `rgb` only",
                             "depth": "true to also return `depth`: float32 (h, w), metres along the optical axis "
                                      "(Z, not ray range). Always sent raw. Default false."})
def capture_image(ctx, robot_id, args):
    """Render an RGB image (and optionally depth) from a camera as it is right now, plus its intrinsics.
    Depth sees whatever is DRAWN in groups 0-2, collision or not: a ghost geom shows up, groups 3+ don't. RGB also draws
    RGB_ONLY_GROUP (4), so a sticker is visible in the image and absent from the depth map.
    Pixels that hit nothing hold ~far (see `depth_range`), not inf: treat depth > 0.99 * far as "no surface"."""
    name = args.get("camera")
    if name not in ctx.cameras:
        raise Rejected(f"unknown camera {name!r}; known: {sorted(ctx.cameras)}")
    cam = ctx.cameras[name]
    encoding = args.get("encoding", "png")
    if encoding not in ("png", "raw"):
        raise Rejected(f"unknown encoding {encoding!r}; use 'png' or 'raw'")
    key = (cam["width"], cam["height"])
    if key not in ctx.renderers:   # first use: build the GL context here, on the physics thread
        ctx.renderers[key] = mujoco.Renderer(ctx.model, cam["height"], cam["width"])
    renderer = ctx.renderers[key]
    rgb_opt = mujoco.MjvOption()               # default shows groups 0-2; the RGB pass also shows the RGB-only group
    rgb_opt.geomgroup[RGB_ONLY_GROUP] = 1
    renderer.update_scene(ctx.data, camera=name, scene_option=rgb_opt)
    out = {"camera": name, "time": ctx.data.time, "intrinsics": cam["intrinsics"],
           "rgb": encode_image(renderer.render(), encoding)}
    if args.get("depth", False):   # same scene, second pass: MuJoCo linearises to metres itself
        extent = ctx.model.stat.extent
        renderer.update_scene(ctx.data, camera=name, scene_option=mujoco.MjvOption())   # groups 0-2 only
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render()
        finally:
            renderer.disable_depth_rendering()
        out["depth"] = encode_image(depth, "raw")
        out["depth_range"] = {"near": ctx.model.vis.map.znear * extent, "far": ctx.model.vis.map.zfar * extent}
    return out


@verb("get_state", args={"frame": "'world' (default) or 'base': frame of the robot's end_effector pose. Ignored for robot_id 'world'. "
                                  "The reply's `frame` field always says which frame the coordinates are in."})
def get_state(ctx, robot_id, args):
    """robot: joint/tool state + end-effector pose.  world: pose of every named free body."""
    m, d = ctx.model, ctx.data
    if robot_id == WORLD:   # `frame` is ignored here: bodies are always reported in the world frame
        bodies = {}
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                b = m.jnt_bodyid[j]
                bodies[m.body(b).name] = {"position": d.xpos[b].tolist(), "orientation": d.xquat[b].tolist()}
        return {"frame": "world", "time": d.time, "bodies": bodies}

    frame = args.get("frame", "world")
    if frame not in FRAMES:
        raise Rejected(f"unknown frame {frame!r}; use one of {FRAMES}")
    r = ctx.robots[robot_id]
    q = lambda ids: [float(d.qpos[m.jnt_qposadr[j]]) for j in ids]
    dq = lambda ids: [float(d.qvel[m.jnt_dofadr[j]]) for j in ids]
    state = {"frame": frame, "time": d.time, "joint_positions": q(r["arm_joints"]),
             "joint_velocities": dq(r["arm_joints"]), "tool_joint_positions": q(r["tool_joints"]), "end_effector": None,
             "gripper_opening": _opening(ctx, r["gripper"]) if r["gripper"] else None}
    if r["ee"]:
        pos, quat = ee_pose(m, d, r["ee"])
        if frame == "base":
            pos, quat = pose_from_world(r["frame_pos"], r["frame_quat"], pos, quat)
        state["end_effector"] = {"position": pos.tolist(), "orientation": quat.tolist()}
    return state


@verb("move_joints", args={"positions": "list[float], radians, one per arm joint",
                           "speed": "max joint speed in rad/s (default 1.0)",
                           "tolerance": "max joint error in rad to count as arrived (default 0.01)",
                           "timeout": "sim seconds before failing (default: distance / speed + 5)"})
def move_joints(ctx, robot_id, args):
    """Move the arm to joint targets. Blocks until it has settled there; fails on timeout."""
    if robot_id == WORLD:
        raise Rejected("move_joints needs a robot_id")
    m, d, r = ctx.model, ctx.data, ctx.robots[robot_id]
    if None in r["arm_actuators"] or not r["arm_joints"]:
        raise Rejected(f"{robot_id} has joints without actuators; cannot command them")
    try:
        target = np.asarray(args["positions"], dtype=float)
        speed, tol = float(args.get("speed", 1.0)), float(args.get("tolerance", 0.01))
    except (KeyError, TypeError, ValueError) as exc:
        raise Rejected(f"bad args: need numeric 'positions' ({exc})")
    if target.shape != (len(r["arm_joints"]),):
        raise Rejected(f"expected {len(r['arm_joints'])} positions, got {target.shape}")
    if speed <= 0 or tol <= 0:
        raise Rejected("'speed' and 'tolerance' must be > 0")
    for q, j in zip(target, r["arm_joints"]):
        lo, hi = m.jnt_range[j]
        if m.jnt_limited[j] and not lo - 1e-9 <= q <= hi + 1e-9:
            raise Rejected(f"{m.joint(j).name}: {q:.3f} outside [{lo:.3f}, {hi:.3f}]")

    acts = r["arm_actuators"]
    qadr = [m.jnt_qposadr[j] for j in r["arm_joints"]]
    # ramp the setpoint from where the controller currently is, so the arm never gets a step input
    dist = float(np.max(np.abs(target - d.ctrl[acts])))
    t_end = d.time + float(args.get("timeout", dist / speed + 5.0))
    max_step = speed * m.opt.timestep
    settle = Settle(m.opt.timestep)
    while True:
        cur = d.ctrl[acts]
        d.ctrl[acts] = cur + np.clip(target - cur, -max_step, max_step)
        yield
        err = float(np.max(np.abs(d.qpos[qadr] - target)))
        still = settle.update(d.qpos[qadr])
        if np.allclose(d.ctrl[acts], target) and err < tol and still:
            return {"joint_positions": d.qpos[qadr].tolist(), "error": err, "time": d.time}
        if d.time > t_end:
            raise TimeoutError(f"not within {tol} rad after the timeout (error {err:.3f} rad)")


@verb("move_to_pose", args={"position": "[x, y, z] metres, in `frame`",
                            "orientation": "[w, x, y, z] unit quaternion, in `frame` (default: keep the current orientation)",
                            "frame": "'world' (default) or 'base' = the pose the robot was placed with (pos/euler in the config); applies to position AND orientation",
                            "speed": "end-effector speed in m/s (default 0.25)",
                            "turn": "rotation speed in rad/s (default 1.0)",
                            "tolerance": "position error in m to count as arrived (default 0.002)",
                            "rot_tolerance": "rotation error in rad to count as arrived (default 0.02)",
                            "timeout": "sim seconds before failing (default: path length / speed + 5)"})
def move_to_pose(ctx, robot_id, args):
    """Move the end effector to a Cartesian pose with inverse kinematics. Blocks until it has settled there."""
    if robot_id == WORLD:
        raise Rejected("move_to_pose needs a robot_id")
    m, d, r = ctx.model, ctx.data, ctx.robots[robot_id]
    if r["ee"] is None:
        raise Rejected(f"{robot_id} has no end effector; set `end_effector:` in its config")
    if None in r["arm_actuators"] or not r["arm_joints"]:
        raise Rejected(f"{robot_id} has joints without actuators; cannot command them")
    try:
        goal_pos = np.asarray(args["position"], dtype=float)
        speed, turn = float(args.get("speed", 0.25)), float(args.get("turn", 1.0))
        tol, rot_tol = float(args.get("tolerance", 0.002)), float(args.get("rot_tolerance", 0.02))
        quat = args.get("orientation")
        goal_quat = None if quat is None else np.asarray(quat, dtype=float)
        frame = args.get("frame", "world")
    except (KeyError, TypeError, ValueError) as exc:
        raise Rejected(f"bad args: need numeric 'position' ({exc})")
    if goal_pos.shape != (3,):
        raise Rejected(f"'position' must be [x, y, z], got shape {goal_pos.shape}")
    if goal_quat is not None:
        if goal_quat.shape != (4,) or np.linalg.norm(goal_quat) < 1e-6:
            raise Rejected("'orientation' must be a non-zero quaternion [w, x, y, z]")
        goal_quat = goal_quat / np.linalg.norm(goal_quat)
    if min(speed, turn, tol, rot_tol) <= 0:
        raise Rejected("'speed', 'turn' and tolerances must be > 0")
    if frame not in FRAMES:
        raise Rejected(f"unknown frame {frame!r}; use one of {FRAMES}")
    if frame == "base":   # express the goal in the world frame: everything below (IK, errors) is world
        goal_pos, goal_quat = pose_to_world(r["frame_pos"], r["frame_quat"], goal_pos, goal_quat)

    ee, joints, acts = r["ee"], r["arm_joints"], r["arm_actuators"]
    # the virtual target starts ON the end effector and walks to the goal; the IK chases it
    tgt_pos, tgt_quat = ee_pose(m, d, ee)
    if goal_quat is None:
        goal_quat = tgt_quat.copy()
    _, dist, angle = pose_error(m, d, ee, goal_pos, goal_quat)
    t_end = d.time + float(args.get("timeout", dist / speed + angle / turn + 5.0))
    settle = Settle(m.opt.timestep)
    qadr = [m.jnt_qposadr[j] for j in joints]
    while True:
        twist = step_target(tgt_pos, tgt_quat, goal_pos, goal_quat, speed, turn, m.opt.timestep)
        ik_step(m, d, ee, joints, acts, tgt_pos, tgt_quat, twist=twist, **r["ik"])
        yield
        _, e_pos, e_rot = pose_error(m, d, ee, goal_pos, goal_quat)
        still = settle.update(d.qpos[qadr])
        if e_pos < tol and e_rot < rot_tol and still:
            pos, quat = ee_pose(m, d, ee)
            return {"position": pos.tolist(), "orientation": quat.tolist(),
                    "position_error": e_pos, "rotation_error": e_rot, "time": d.time}
        if d.time > t_end:
            raise TimeoutError(f"not at the pose after the timeout (position error {e_pos:.4f} m, "
                               f"rotation error {e_rot:.3f} rad); unreachable or blocked?")

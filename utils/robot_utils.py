"""Robot maths shared by the verbs: end-effector pose and inverse kinematics.

IK is the one from speedrun12: damped least squares, ONE small step per physics step.
    error (6) = [target pos - ee pos, rotation from ee to target]
    J (6 x n) = how the ee moves per arm joint (mj_jac*), arm dofs only
    dq        = J^T (J J^T + damping^2 I)^-1 error      ->  ctrl = qpos + dq
The IK only ever chases a target a few mm away: `step_target` walks that target to the goal in a
straight line at a bounded speed, so the end effector follows the line instead of swinging in
joint space. Quaternions are MuJoCo's (w, x, y, z); everything is in the world frame.

`ee` is (mjtObj.mjOBJ_SITE | mjOBJ_BODY, id), as built by engine_utils.robot_index.
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

SITE = mujoco.mjtObj.mjOBJ_SITE


class Settle:
    """Settled = the joints stayed within `tol` rad (peak-to-peak) over the last `window` seconds.
    Judged on POSITIONS, not qvel: a force-saturated servo can chatter +-0.1 rad/s every single step (see
    ik_step's `lead`), and one qvel sample aliases that into a constant. Call update() on every step."""

    def __init__(self, dt, window=0.1, tol=2e-3):
        self.tol, self.buf = tol, deque(maxlen=max(2, int(window / dt)))

    def update(self, q) -> bool:
        self.buf.append(np.array(q, dtype=float))
        return len(self.buf) == self.buf.maxlen and float(np.ptp(np.array(self.buf), axis=0).max()) < self.tol


def ee_pose(model, data, ee):
    """(position, quaternion) of the end effector in the world frame."""
    kind, i = ee
    pos, mat = (data.site_xpos[i], data.site_xmat[i]) if kind == SITE else (data.xpos[i], data.xmat[i])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return pos.copy(), quat


def pose_to_world(frame_pos, frame_quat, pos, quat=None):
    """A pose given in a robot's frame -> world. The robot's frame is the pose it was placed with."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, frame_quat)
    world_quat = None
    if quat is not None:
        world_quat = np.zeros(4)
        mujoco.mju_mulQuat(world_quat, frame_quat, quat)
    return frame_pos + rot.reshape(3, 3) @ pos, world_quat


def pose_from_world(frame_pos, frame_quat, pos, quat):
    """A world pose -> the robot's frame (inverse of pose_to_world)."""
    rot, inv, local_quat = np.zeros(9), np.zeros(4), np.zeros(4)
    mujoco.mju_quat2Mat(rot, frame_quat)
    mujoco.mju_negQuat(inv, frame_quat)
    mujoco.mju_mulQuat(local_quat, inv, quat)
    return rot.reshape(3, 3).T @ (pos - frame_pos), local_quat


def pose_error(model, data, ee, target_pos, target_quat):
    """(6-vector error in the world frame, position error in m, rotation error in rad)."""
    pos, quat = ee_pose(model, data, ee)
    err, rot = np.zeros(6), np.zeros(3)
    err[:3] = target_pos - pos
    mujoco.mju_subQuat(rot, target_quat, quat)       # rotation error in the ee frame...
    mujoco.mju_rotVecQuat(err[3:], rot, quat)        # ...the Jacobian is in the world frame -> rotate it
    return err, float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))


def step_target(pos, quat, goal_pos, goal_quat, speed, turn, dt):
    """Advance the virtual target one dt toward the goal, IN PLACE: straight line at <= speed m/s,
    shortest rotation at <= turn rad/s. Returns the target's twist [linear, angular] in the world
    frame, for the IK's velocity feedforward."""
    twist, w = np.zeros(6), np.zeros(3)
    step = goal_pos - pos
    twist[:3] = step * min(1 / dt, speed / (np.linalg.norm(step) + 1e-9))
    pos += twist[:3] * dt
    mujoco.mju_subQuat(w, goal_quat, quat)
    w *= min(1.0, turn * dt / (np.linalg.norm(w) + 1e-9))
    mujoco.mju_rotVecQuat(twist[3:], w / dt, quat)    # local -> world
    mujoco.mju_quatIntegrate(quat, w, 1)
    return twist


def ik_step(model, data, ee, joints, acts, target_pos, target_quat, damping=0.05, max_dq=1.0, lead=0.0, twist=None):
    """One damped-least-squares step: writes ctrl[acts] = qpos + dq for the arm `joints`.

    Damping keeps dq finite near a singularity; for an unreachable target the arm stretches
    toward it and stays there. max_dq is only a guard, since the target can't jump.
    lead = velocity feedforward in seconds: aim at where the target will be in `lead` s (err += lead * twist).
    A servo with force = kp (ctrl - q) - kv qdot, kv = kp / 10, lags a moving joint by ~0.1 s, so the Panda
    uses 0.1 (= kv / kp). It is 0 on the UR10e: 0.1 saturated its light wrist_3 and made it flip every step.
    """
    dofs = [model.jnt_dofadr[j] for j in joints]
    qadr = [model.jnt_qposadr[j] for j in joints]
    err, _, _ = pose_error(model, data, ee, target_pos, target_quat)
    if lead and twist is not None:
        err += lead * twist
    J = np.zeros((6, model.nv))
    (mujoco.mj_jacSite if ee[0] == SITE else mujoco.mj_jacBody)(model, data, J[:3], J[3:], ee[1])
    J = J[:, dofs]                                    # nv columns include free objects: keep the arm
    dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), err)
    q = data.qpos[qadr] + np.clip(dq, -max_dq, max_dq)
    for k, j in enumerate(joints):                    # stay inside the joint limits
        if model.jnt_limited[j]:
            q[k] = np.clip(q[k], *model.jnt_range[j])
    data.ctrl[acts] = q

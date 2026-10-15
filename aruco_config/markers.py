"""ArUco markers: the scene config's `markers:` list -> textured stickers in the MuJoCo scene.

Same syntax as the PyBullet template:

    markers:
      - name: ref_a
        id: 20
        dictionary: DICT_4X4_50
        size: 0.05                  # side of the BLACK square as printed, metres
        margin: 0.01                # white quiet zone around it (a tag without one may not decode)
        reference: true             # world-fixed anchor for camera calibration (parsed, not used yet)
        position: [0.2, 0.35, 0.001]
        orientation: [0, 0, 0]      # roll, pitch, yaw in radians; [0, 0, 0] = lying face-up
        attach_to: {body: cube}     # optional: position/orientation are then in that body's frame

Marker frame = the ArUco convention: origin at the centre, +X right, +Y up, +Z out of the face.
A sticker is a 1 mm slab in RGB_ONLY_GROUP with no collision: drawn in the image, absent from the depth map
and untouchable, like printed paper. (A slab, not a plane: MuJoCo only allows planes in the static world body,
and a marker must be able to ride on a moving object.)

Print the markers of a config:   python aruco_config/markers.py config/ur10e_robotiq.yaml out/
"""
import os
import sys

import cv2
import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.camera_utils import RGB_ONLY_GROUP

HALF_THICKNESS = 0.0005


def marker_image(m: dict, side: int = 480) -> np.ndarray:
    """Greyscale image of the marker plus its white quiet zone: (side + 2 * pad) pixels square."""
    name = m.get("dictionary", "DICT_4X4_50")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"marker {m['name']!r}: unknown dictionary {name!r}")
    try:
        img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), m["id"], side)
    except cv2.error as exc:
        raise ValueError(f"marker {m['name']!r}: id {m['id']} is not valid in {name}") from exc
    pad = round(m.get("margin", 0.01) / m["size"] * side)
    return np.pad(img, pad, constant_values=255)


def add_markers(scene: mujoco.MjSpec, cfg: dict) -> None:
    for m in cfg.get("markers", []):
        img = marker_image(m)
        half = m["size"] / 2 + m.get("margin", 0.01)
        tex = scene.add_texture(name=f"{m['name']}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                width=img.shape[1], height=img.shape[0], nchannel=3)
        tex.data = np.repeat(img[..., None], 3, axis=2).tobytes()   # as-is: renders un-mirrored, +X right / +Y up
        mat = scene.add_material(name=f"{m['name']}_mat", specular=0.0, shininess=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        quat, down = np.zeros(4), np.zeros(3)
        mujoco.mju_euler2Quat(quat, m.get("orientation", [0, 0, 0]), "XYZ")   # roll, pitch, yaw about fixed axes
        mujoco.mju_rotVecQuat(down, [0, 0, -HALF_THICKNESS], quat)   # slab centre sits below the face: the marker frame is ON the face
        parent = scene.worldbody
        if "attach_to" in m:
            body = m["attach_to"]["body"]
            parent = next((b for b in scene.bodies if b.name == body), None)
            if parent is None:
                raise ValueError(f"marker {m['name']!r}: attach_to body {body!r} not found")
        parent.add_geom(name=m["name"], type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, HALF_THICKNESS],
                        pos=np.asarray(m["position"]) + down, quat=quat, material=mat.name, group=RGB_ONLY_GROUP,
                        contype=0, conaffinity=0)


if __name__ == "__main__":
    import yaml
    cfg, out = yaml.safe_load(open(sys.argv[1])), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for m in cfg.get("markers", []):
        path = os.path.join(out, f"{m['name']}_{m.get('dictionary', 'DICT_4X4_50')}_id{m['id']}.png")
        cv2.imwrite(path, marker_image(m, 600))
        print(f"{path}   print so the BLACK square is {m['size'] * 1000:g} mm, white border {m.get('margin', 0.01) * 1000:g} mm")

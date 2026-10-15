"""Cameras: place them from a look-at point, describe their intrinsics, encode their images.

A MuJoCo camera looks along its own -z, with x to the right and y up. Poses are NEVER sent to the
client: like a real calibrated camera, a client only gets the image and the intrinsics, and has to work
out where the camera is (ArUco markers) on its own.
"""
from __future__ import annotations

import base64
import io

import mujoco
import numpy as np

RGB_ONLY_GROUP = 4   # geom group drawn in the RGB pass but NOT in the depth pass (ArUco stickers: printed paper has no depth).
                     # 0-2 = seen by both, 3 = hidden from both (collision pieces); groups are only a rendering switch


def look_at_quat(pos, lookat, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Orientation quaternion (w, x, y, z) of a camera at `pos` looking at `lookat`."""
    forward = np.asarray(lookat, dtype=float) - np.asarray(pos, dtype=float)
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("camera pos and lookat are the same point")
    forward /= np.linalg.norm(forward)
    if abs(forward @ up) > 0.999:          # looking straight up/down: world z can't define "up"
        up = (0.0, 1.0, 0.0)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    mat = np.column_stack([right, true_up, -forward])   # camera axes in world coordinates
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat


def intrinsics(width: int, height: int, fovy_deg: float) -> dict:
    """Pinhole intrinsics in pixels. MuJoCo pixels are square and the principal point is the image centre."""
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    return {"fx": float(f), "fy": float(f), "cx": width / 2, "cy": height / 2, "width": width, "height": height}


def encode_image(img: np.ndarray, encoding: str = "png") -> dict:
    """Image payload: {"encoding": "png" | "raw", "shape", "dtype", "data": base64}. png is lossless."""
    if encoding == "raw":
        data = img.tobytes()
    elif encoding == "png":
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        data = buf.getvalue()
    else:
        raise ValueError(f"unknown encoding {encoding!r}; use 'png' or 'raw'")
    return {"encoding": encoding, "shape": list(img.shape), "dtype": str(img.dtype),
            "data": base64.b64encode(data).decode("ascii")}

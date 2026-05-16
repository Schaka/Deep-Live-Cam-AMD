"""Shared face-mask utilities.

build_face_hull_mask is used by both face_swapper and the GPEN face enhancer.
It lives here so neither module owns it and both share the same cache.
"""

from typing import Any

import cv2
import numpy as np


# Single-slot cache keyed on sampled landmark positions + frame dimensions.
# Within one frame both the swapper and enhancer see the same landmark_2d_106
# data, so the second call is a free dict lookup.
_hull_cache: dict = {'key': None, 'mask': None}


def build_face_hull_mask(face: Any, frame_shape: tuple) -> "np.ndarray | None":
    """Build a feathered convex-hull face mask from landmark_2d_106 in frame space.

    Returns float32 [0, 1] or None when landmarks are unavailable.
    GaussianBlur runs only on a tight crop around the hull bounding box.
    Cached: same landmark positions in the same frame → free lookup.
    """
    lm = getattr(face, 'landmark_2d_106', None)
    if lm is None or not isinstance(lm, np.ndarray) or lm.shape[0] < 106:
        return None
    if not np.all(np.isfinite(lm)):
        return None

    # Cache key: every 5th landmark rounded to nearest pixel + frame dimensions.
    lm_key = (frame_shape[:2], lm[::5].round(0).astype(np.int16).tobytes())
    if _hull_cache['key'] == lm_key:
        return _hull_cache['mask']

    h, w = frame_shape[:2]
    lm_int = lm.astype(np.int32)

    face_outline = lm_int[0:33]
    eyebrows = lm_int[33:43]
    if len(eyebrows) > 0:
        chin = lm_int[16].astype(np.float32)
        eyebrow_center = np.mean(eyebrows.astype(np.float32), axis=0)
        up_vec = eyebrow_center - chin
        norm = float(np.linalg.norm(up_vec))
        if norm > 0:
            up_vec /= norm
            forehead_pts = eyebrows.astype(np.float32) + up_vec * norm * 1.0
            top_center = np.mean(forehead_pts, axis=0)
            forehead_pts = (forehead_pts - top_center) * 1.2 + top_center
            face_outline = np.concatenate(
                (face_outline, forehead_pts.astype(np.int32)), axis=0
            )

    try:
        hull = cv2.convexHull(face_outline.astype(np.float32))
        if hull is None or len(hull) < 3:
            _hull_cache['key'] = lm_key
            _hull_cache['mask'] = None
            return None
        hull_int = hull.astype(np.int32)
    except Exception:
        return None

    _PAD = 35  # >= blur radius (31/2 = 15) + margin
    xs, ys = hull_int[:, 0, 0], hull_int[:, 0, 1]
    x1 = max(0, int(xs.min()) - _PAD)
    y1 = max(0, int(ys.min()) - _PAD)
    x2 = min(w, int(xs.max()) + _PAD)
    y2 = min(h, int(ys.max()) + _PAD)
    if x1 >= x2 or y1 >= y2:
        _hull_cache['key'] = lm_key
        _hull_cache['mask'] = None
        return None

    crop_hull = hull_int.copy()
    crop_hull[:, 0, 0] -= x1
    crop_hull[:, 0, 1] -= y1
    small = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillConvexPoly(small, crop_hull, 255)
    small = cv2.GaussianBlur(small, (31, 31), 0)

    full = np.zeros((h, w), dtype=np.float32)
    full[y1:y2, x1:x2] = small.astype(np.float32) * (1.0 / 255.0)

    _hull_cache['key'] = lm_key
    _hull_cache['mask'] = full
    return full

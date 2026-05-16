"""GPEN-BFR-512 face enhancer — ONNX-based face restoration at 512x512."""

from typing import Any, List
import os
import threading

import cv2
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import (
    is_image,
    is_video,
)
from insightface.app.common import Face as _InsightFace
from modules.processors.frame._onnx_enhancer import (
    KpsEma,
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
)

NAME = "DLC.FACE-ENHANCER-GPEN512"
INPUT_SIZE = 512
MODEL_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx"
MODEL_FILE = "gpen_bfr_512.onnx"
ENHANCE_BLEND = 0.80  # 1.0 = full GPEN output; lower preserves original texture

ENHANCER = None
ENHANCER_FAILED = False
THREAD_LOCK = threading.Lock()
_kps_ema = KpsEma()

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    model_path = os.path.join(models_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        update_status(f"Downloading {MODEL_FILE}...", NAME)
        from modules.utilities import conditional_download
        conditional_download(models_dir, [MODEL_URL])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_enhancer() -> Any:
    global ENHANCER, ENHANCER_FAILED
    if ENHANCER_FAILED:
        raise RuntimeError(f"{NAME}: Model failed to load (see earlier log).")
    with THREAD_LOCK:
        if ENHANCER is None and not ENHANCER_FAILED:
            try:
                model_path = os.path.join(models_dir, MODEL_FILE)
                if not os.path.exists(model_path):
                    from modules.utilities import conditional_download
                    conditional_download(models_dir, [MODEL_URL])
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                print(f"{NAME}: Loading ONNX model from {model_path}")
                ENHANCER = create_onnx_session(model_path)
                warmup_session(ENHANCER)
                print(f"{NAME}: Model loaded successfully.")
            except Exception:
                ENHANCER_FAILED = True
                raise
    return ENHANCER


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        if face.kps is not None:
            smoothed = _kps_ema.smooth(face.kps.astype(np.float32))
            face = _InsightFace({**face, 'kps': smoothed})
        return enhance_face_onnx(temp_frame, face, session, INPUT_SIZE,
                                  blend_strength=ENHANCE_BLEND)
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def enhance_face_inline(temp_frame: Frame, face: Face) -> Frame:
    """Enhance without EMA — called inline from the swap pipeline so both
    operations use the exact same keypoints and geometry."""
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        return enhance_face_onnx(temp_frame, face, session, INPUT_SIZE,
                                  blend_strength=ENHANCE_BLEND)
    except Exception as e:
        print(f"{NAME}: Error during inline face enhancement: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    if detected_faces:
        target_face = detected_faces[0]
    else:
        target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

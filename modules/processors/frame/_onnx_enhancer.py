"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.face_mask import build_face_hull_mask

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))


class KpsEma:
    """Adaptive-alpha EMA smoother for 5-point face keypoints.

    At rest (dist ≈ 0): uses alpha_min — heavy smoothing, no idle jitter.
    When moving fast (dist → jump_threshold/2): ramps to alpha_max — responsive,
    no lag. Resets entirely on jumps larger than jump_threshold.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        alpha_max: float = 0.92,
        jump_threshold: float = 25.0,
    ) -> None:
        self._alpha_min = alpha
        self._alpha_max = alpha_max
        self._jump = jump_threshold
        self._prev: "np.ndarray | None" = None
        self._last_t: float = 0.0

    def smooth(self, kps: np.ndarray) -> np.ndarray:
        kps = kps.astype(np.float32)
        now = time.perf_counter()
        if self._prev is not None:
            dist = float(np.linalg.norm(kps - self._prev))
            if dist > self._jump:
                self._prev = kps.copy()
            else:
                # Scale alpha_max by how much time has passed relative to 30 fps.
                # Sub-30 fps → longer interval → legitimate fast motion displaces
                # more pixels per frame, so be proportionally more responsive.
                dt = now - self._last_t
                time_scale = min(dt / (1.0 / 30.0), 1.5) if self._last_t else 1.0
                alpha_max_t = min(self._alpha_max * time_scale, 1.0)
                t = min(dist / (self._jump * 0.30), 1.0)
                alpha = self._alpha_min + t * (alpha_max_t - self._alpha_min)
                kps = alpha * kps + (1.0 - alpha) * self._prev
        self._prev = kps.copy()
        self._last_t = now
        return kps

    def reset(self) -> None:
        self._prev = None
        self._last_t = 0.0


def _is_gpu_provider_active(providers=None) -> bool:
    if providers is None:
        providers = modules.globals.execution_providers
    gpu_prefixes = ("CUDA", "MIGraphX", "ROCM", "Dml", "CoreML")
    for p in providers:
        name = p[0] if isinstance(p, tuple) else p
        if any(name.startswith(pfx) for pfx in gpu_prefixes):
            return True
    return False


def make_session_options(providers=None) -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    if _is_gpu_provider_active(providers):
        n = max(1, modules.globals.execution_threads)
        opts.inter_op_num_threads = n
        opts.intra_op_num_threads = n
    return opts


def build_provider_config(providers=None):
    """Wrap raw provider name strings with optimised CUDA / CoreML options.

    Providers that are already ``(name, options_dict)`` tuples are passed
    through unchanged.  Non-CUDA providers are left as bare strings.
    """
    if providers is None:
        providers = modules.globals.execution_providers

    config = []
    for p in providers:
        if isinstance(p, tuple):
            # Already configured – pass through
            config.append(p)
        elif p == "CUDAExecutionProvider":
            # Use bare provider — ONNX Runtime's defaults are fastest on
            # modern GPUs (Blackwell/sm_120).  Custom options like
            # EXHAUSTIVE cudnn_conv_algo_search hurt performance on these
            # architectures.
            config.append(p)
        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                },
            ))
        elif p == "MIGraphXExecutionProvider":
            _cache_dir = os.path.expanduser("~/.cache/migraphx_models")
            os.makedirs(_cache_dir, exist_ok=True)
            config.append((
                "MIGraphXExecutionProvider",
                {
                    "migraphx_fp16_enable": "1",
                    "migraphx_model_cache_dir": _cache_dir,
                },
            ))
        else:
            config.append(p)
    return config


def run_inference(session: onnxruntime.InferenceSession,
                  input_name: str,
                  input_tensor: "np.ndarray") -> "np.ndarray":
    """Run ONNX inference, using IO binding when a CUDA session is active.

    IO binding avoids redundant host↔device copies by transferring the
    input tensor directly to GPU memory and letting ONNX Runtime allocate
    the output on the device.  Falls back to the standard ``session.run``
    path for non-CUDA providers or if binding fails.
    """
    if "CUDAExecutionProvider" in session.get_providers():
        try:
            io_binding = session.io_binding()

            # Input: numpy → GPU
            ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_tensor, "cuda", 0,
            )
            io_binding.bind_ortvalue_input(input_name, ort_input)

            # Output: allocate on GPU (avoids a CPU-side allocation)
            output_name = session.get_outputs()[0].name
            io_binding.bind_output(output_name, "cuda", 0)

            session.run_with_iobinding(io_binding)

            return io_binding.get_outputs()[0].numpy()
        except Exception:
            # Fall back to standard path (e.g. ORT version mismatch,
            # unsupported op, or VRAM pressure)
            pass

    return session.run(None, {input_name: input_tensor})[0]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with optimised provider config.

    On Apple Silicon, applies CoreML graph optimizations (Pad decomposition,
    Shape/Gather folding, Split decomposition) to reduce CPU↔ANE partition
    boundaries.

    On AMD/MiGraphX, falls back fp16 → fp32 → CPU when the compiler can't
    handle the model graph (std::bad_alloc in migraphx_parse_onnx_buffer).
    """
    if IS_APPLE_SILICON:
        from modules.onnx_optimize import optimize_for_coreml
        # Infer input shape from the model for Shape/Gather folding
        try:
            import onnx
            m = onnx.load(model_path)
            inp = m.graph.input[0]
            dims = inp.type.tensor_type.shape.dim
            shape = tuple(d.dim_value for d in dims if d.dim_value > 0)
            input_shape = shape if len(shape) == 4 else None
        except Exception:
            input_shape = None
        model_path = optimize_for_coreml(model_path, input_shape=input_shape)

    providers = build_provider_config()
    # GPEN models are FP16 — add FP16 flag for MIGraphX here only.
    # build_provider_config is also used by retinaface detection where
    # FP16 causes NaN in NMS, so the flag lives here instead.
    providers = [
        (p[0], {**p[1], "migraphx_fp16_enable": "1"})
        if isinstance(p, tuple) and p[0] == "MIGraphXExecutionProvider"
        else p
        for p in providers
    ]
    session_options = make_session_options(providers)

    try:
        session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers,
        )
        return session
    except Exception as e:
        print(f"ONNX enhancer: Primary providers failed: {e}")

    # MiGraphX fp16 failed — retry fp32 (avoids graph-expansion OOM in compiler)
    _has_migraphx = any(
        (isinstance(p, tuple) and p[0] == "MIGraphXExecutionProvider")
        or p == "MIGraphXExecutionProvider"
        for p in providers
    )
    if _has_migraphx:
        _cache_dir = os.path.expanduser("~/.cache/migraphx_models")
        fp32_providers = [
            ("MIGraphXExecutionProvider", {
                "migraphx_fp16_enable": "0",
                "migraphx_model_cache_dir": _cache_dir,
            }) if (isinstance(p, tuple) and p[0] == "MIGraphXExecutionProvider")
            or p == "MIGraphXExecutionProvider"
            else p
            for p in providers
        ]
        try:
            session = onnxruntime.InferenceSession(
                model_path, sess_options=session_options, providers=fp32_providers,
            )
            print("ONNX enhancer: Loaded with MiGraphX fp32.")
            return session
        except Exception as e:
            print(f"ONNX enhancer: MiGraphX fp32 also failed: {e}")

    # CPU fallback — model loads, just slower
    print("ONNX enhancer: All GPU providers failed. Falling back to CPUExecutionProvider.")
    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=["CPUExecutionProvider"],
    )
    print("ONNX enhancer: Loaded with CPUExecutionProvider.")
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
    blend_strength: float = 1.0,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model.

    blend_strength: 1.0 = full enhanced output, <1.0 blends back original
    texture to reduce over-smoothing from aggressive face restoration models.
    Uses landmark_2d_106 convex-hull mask when available to avoid the visible
    rectangular boundary that the warped rectangular mask produces.
    """
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        input_name = session.get_inputs()[0].name
        output = run_inference(session, input_name, blob)
    enhanced = postprocess_face(output)

    # Rectangular feathered mask in aligned space — prevents warp-edge seam artifacts
    rect_mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(1, input_size // 16)
    rect_mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    rect_mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    rect_mask[:, :border] = np.minimum(rect_mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    rect_mask[:, -border:] = np.minimum(rect_mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    h, w = frame.shape[:2]
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_rect = cv2.warpAffine(
        rect_mask, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    # Intersect rectangular warp mask with face-contour hull mask.
    # Hull mask clips blend to face shape (no visible box against backgrounds).
    # Rectangular mask clips to the warped GPEN output region (no seam at warp edge).
    hull_mask = build_face_hull_mask(face, frame.shape)
    if hull_mask is not None:
        blend_mask = np.minimum(warped_rect, hull_mask)
    else:
        blend_mask = warped_rect

    mask_3ch = blend_mask[:, :, np.newaxis] * blend_strength
    result = (warped_enhanced.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)

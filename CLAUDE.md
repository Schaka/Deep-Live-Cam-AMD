# Deep-Live-Cam (AMD Fork) - Project Context

## What This Is

Fork of hacksider/Deep-Live-Cam focused on AMD GPU support via ROCm 7.0+ and the MiGraphX ONNX Runtime execution provider. The upstream project moved to a paid model while keeping the AGPLv3 repo stagnant. This fork exists for AMD users.

## Key Architecture

```
_CaptureWorker  ->  capture_queue (maxsize=2, drop-oldest on full)
                         |
_ProcessingWorker  (queue drain in async mode -> always newest frame)
    detect_one_face_with_landmarks()  ->  kps + landmark_2d_106
    swap_face()
        paste_back=False  ->  bgr_fake + M
        _face_mask_paste_back()  ->  landmark convex-hull blend
                         |
               processed_queue (maxsize=2)
                         |
                      Display
```

## Hardware Target

- **Tested GPU:** RX 9070 XT
- **ROCm:** 7.0+
- **Provider:** MiGraphX (`--execution-provider migraphx`)
- **Python:** 3.12
- **Model:** `inswapper_128_fp16.onnx` (FP16 required, not FP32)

## Philosophy: GPU-First

All inference must run on the GPU. The CPU should be nearly idle during normal operation. This requires:

- `HSA_ENABLE_INTERRUPT=1` (interrupt mode, not spin-polling -- this is the critical one)
- `GPU_MAX_HW_QUEUES=1`
- Thread counts for OMP/MKL/OpenBLAS/GOTO set to 1
- ONNX Runtime `SessionOptions` with `inter_op_num_threads=1`, `intra_op_num_threads=1`
- `cv2.setNumThreads(1)` in the processing worker
- All set automatically in `run.py` when MiGraphX is detected

## Key Files

| File | Role |
|---|---|
| `run.py` | Entry point, sets ROCm env vars before imports |
| `modules/processors/frame/face_swapper.py` | Face swap, mask blending, MiGraphX provider config |
| `modules/processors/frame/face_analyser.py` | Face detection with landmark_2d_106 |
| `modules/processors/frame/_onnx_enhancer.py` | GPEN/GFPGAN enhancer, shared SessionOptions helper |
| `modules/ui.py` | PyQt UI, _ProcessingWorker, _CaptureWorker, sync mode |
| `modules/globals.py` | Global state including `live_sync` flag |
| `jitter_handling.md` | Detailed write-up of jitter root causes and fixes |
| `run-on-linux.md` | AMD Linux setup instructions |

## Key Functions

| Function | File | Purpose |
|---|---|---|
| `detect_one_face_with_landmarks` | `face_analyser.py` | Detection + landmark_2d_106, no recognition/genderage |
| `_face_mask_paste_back` | `face_swapper.py` | Blend using face-contour convex-hull mask |
| `create_face_mask` | `face_swapper.py` | Build feathered mask from landmark_2d_106 |
| `_fast_paste_back` | `face_swapper.py` | Fallback: feathered rectangular affine mask |
| `make_session_options` | `_onnx_enhancer.py` | Shared ORT SessionOptions with thread limits |
| `get_face_swapper` | `face_swapper.py` | Loads inswapper with MiGraphX FP16 config |

## Globals That Matter

| Global | Default | Effect |
|---|---|---|
| `live_sync` | `False` | True = process every frame in order, no queue drain, no stabilizer |

## MiGraphX Provider Config

```python
("MIGraphXExecutionProvider", {
    "migraphx_fp16_enable": "1",
    "migraphx_model_cache_dir": "~/.cache/migraphx_models",
})
```

## Do Not Break

- `HSA_ENABLE_INTERRUPT=1` must be set before any ROCm/HIP import. It is set in `run.py` at the very top.
- `detect_one_face_with_landmarks` (not `detect_one_face_fast`) must be used in the live loop so `landmark_2d_106` is always present.
- `_face_mask_paste_back` must fall back to `_fast_paste_back` when landmarks are absent or invalid, never crash.
- Sync mode must disable both queue draining and the dead-zone stabilizer. These two are coupled -- enabling one without the other causes misalignment.

## Models

- `models/inswapper_128_fp16.onnx`: https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx
- `models/GFPGANv1.4.onnx`: https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx

## Setup (Short Version)

```bash
distrobox create --name dlc-amd --image rocm/onnxruntime:rocm7.2.3_ub24.04_ort1.23_torch2.10.0
distrobox enter dlc-amd
python3 -m venv venv && source venv/bin/activate
pip uninstall onnxruntime onnxruntime-gpu -y
pip install -r requirements.txt
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
python3 run.py --execution-provider migraphx
```

# Jitter Handling — Findings & Solutions

## Root Causes Identified

### 1. Visible rectangular boundary around swapped face
**Cause:** Both insightface's `paste_back=True` and the custom `_fast_paste_back` use the rectangular
affine boundary of the 128×128 inswapper crop warped back to output space. Against high-contrast
backgrounds (e.g. a door frame), this quadrilateral edge is visible — it does not follow the face shape.

**Fix:** `_face_mask_paste_back()` in `face_swapper.py`. Uses `landmark_2d_106` to build a feathered
convex-hull face mask (`create_face_mask()`), blending along the natural face/background boundary
instead of the rectangular crop edge. Requires landmarks on the target face object.

Detection was switched from `detect_one_face_fast` (det model only) to `detect_one_face_with_landmarks`
(det + `landmark_2d_106`, skips recognition/genderage) so landmarks are always available in the live loop.
`swap_face()` falls back to `_fast_paste_back` if landmarks are absent.

---

### 2. Swap position lagging during fast movement (async mode)
**Cause:** The capture queue (`maxsize=2`) fills up when processing is slower than the webcam framerate.
The ProcessingWorker would process frame N while frames N+1, N+2 had already arrived. By display time,
the swap was placed at a position 1–2 frames old. The face-shaped mask was drawn from the correct
(current) landmarks, but the swap kps was stale → swap and mask misaligned → visible artifact on movement.

**Fix:** Queue drain in `_ProcessingWorker.run()`. When `live_sync=False` (default), after taking a
frame from the capture queue, drain any remaining frames with `get_nowait()` and keep only the newest.
Bounds display lag to one capture interval (~33ms at 30fps) regardless of processing speed.

---

### 3. Stabilization lag causing swap/mask misalignment in sync mode
**Cause:** The dead-zone stabilizer (`_NOISE_FLOOR=0.5`, `_CONVERGE_CEIL=2.5`) blends kps toward the
new position with alpha 0.5–0.95 during fast movement. In async mode this smooths out stale-frame
jumps. In sync mode it's counterproductive: frames are fresh and consecutive, so the face moves
smoothly anyway. The alpha < 1.0 means kps lags the true position, while `landmark_2d_106` is always
accurate (from the current frame). Swap placed at lagging kps, mask drawn from accurate landmarks →
misalignment → jitter during fast movement.

**Fix:** Stabilization is skipped entirely when `live_sync=True`. Raw detection passes directly to
`swap_face()`. Stabilization still runs in async mode where it is needed.

---

## Architecture

```
_CaptureWorker  →  capture_queue (maxsize=2, drop-oldest on full)
                         ↓
_ProcessingWorker  (queue drain in async mode → always newest frame)
    detect_one_face_with_landmarks()  →  kps + landmark_2d_106
    swap_face()
        paste_back=False  →  bgr_fake + M
        _face_mask_paste_back()  →  landmark convex-hull blend
                         ↓
               processed_queue (maxsize=2)
                         ↓
                      Display
```

## Key Functions

| Function | File | Purpose |
|---|---|---|
| `detect_one_face_with_landmarks` | `face_analyser.py` | Det + landmark_2d_106, no recognition |
| `_face_mask_paste_back` | `face_swapper.py` | Blend using face-contour mask |
| `create_face_mask` | `face_swapper.py` | Feathered convex-hull mask from landmark_2d_106 |
| `_fast_paste_back` | `face_swapper.py` | Fallback: feathered rectangular affine mask |
| `_get_soft_alpha` | `face_swapper.py` | Cached feathered 128×128 alpha template |

## Globals

| Global | Default | Effect |
|---|---|---|
| `live_sync` | `True` | Process frames in FIFO order, no queue drain, no stabilization |

## Hardware Notes (RX 9070 XT / MIGraphX)

- MIGraphX achieves ~30 face swaps/second on this GPU — matches webcam framerate.
- The FPS counter in `_ProcessingWorker` measures actual completed swap iterations, not display rate.
- FP16 is not the cause of any quality issues (confirmed).
- MIGraphX model cache at `~/.cache/migraphx_models/` avoids recompilation on startup.
- `migraphx_fp16_enable=1` is set in the provider config for `get_face_swapper()`.

---

### 4. All remaining jitter resolved by CPU load reduction

**Cause:** ROCm/HIP defaults to busy-polling CPU threads to detect GPU completion instead of
using hardware interrupts. With all CPU cores saturated (100% across all cores), the
`_ProcessingWorker` and `_CaptureWorker` threads were starved of CPU time, causing irregular
frame processing intervals. This produced inconsistent swap cadence even when queue draining
and stabilization were correctly implemented — the timing variance itself was the jitter source.

**Fix:** Set the following environment variables in `run.py` before any imports, when MIGraphX
is detected in `sys.argv`:

```python
os.environ.setdefault('HSA_ENABLE_INTERRUPT', '1')   # interrupt mode instead of spin-polling
os.environ.setdefault('GPU_MAX_HW_QUEUES', '1')       # one HIP queue per device instead of 8+
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('GOTO_NUM_THREADS', '1')
```

Additionally:
- `inter_op_num_threads=1` and `intra_op_num_threads=1` on all ORT `SessionOptions` (via
  `make_session_options()` in `_onnx_enhancer.py`), applied to face swapper, face analyser,
  face enhancer, and GPEN sessions.
- `cv2.setNumThreads(1)` called at the start of `_ProcessingWorker.run()` when MIGraphX is active.

`HSA_ENABLE_INTERRUPT=1` is the decisive fix. The symptom — all cores at 100%, all showing
as separate `python3 run.py --execution-provider migraphx` processes in htop — is the classic
HIP spin-poll signature on ROCm. Interrupts reduce idle CPU to near zero, giving the worker
threads consistent scheduling and eliminating the timing jitter entirely.

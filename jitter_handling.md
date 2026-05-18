# Jitter Handling — Findings & Solutions

## Root Causes & Fixes

### 1. Visible rectangular boundary around swapped face
**Cause:** Both insightface's `paste_back=True` and the custom `_fast_paste_back` use the rectangular
affine boundary of the 128×128 inswapper crop warped back to output space. Against high-contrast
backgrounds this quadrilateral edge is visible — it does not follow the face shape.

**Fix:** `_face_mask_paste_back()` in `face_swapper.py`. Uses `landmark_2d_106` to build a feathered
convex-hull face mask (`create_face_mask()`), blending along the natural face/background boundary.
Detection switched from `detect_one_face_fast` to `detect_one_face_with_landmarks` so landmarks are
always available in the live loop. Falls back to `_fast_paste_back` if landmarks are absent.

---

### 2. Swap position lagging / stale-frame jumps
**Cause:** The capture queue (`maxsize=2`) fills when processing is slower than the webcam framerate.
The ProcessingWorker processes frame N while frames N+1, N+2 have already arrived. By display time
the swap is at a position 1–2 frames old; the face-shaped mask is drawn from current landmarks but
the swap kps is stale → misalignment on movement.

**Fix:** Queue drain in `_ProcessingWorker.run()`. After taking a frame from the capture queue, drain
any remaining frames with `get_nowait()` and keep only the newest. Bounds display lag to one capture
interval (~33ms at 30fps) regardless of processing speed.

---

### 3. Sub-pixel kps noise causing swap/mask/enhancer position jitter
**Cause:** ONNX detection produces sub-pixel noise (~0.3–0.5px) on every call even for a still face.
Without stabilization, the swap affine (M), the hull mask boundary, and the face enhancer paste
position all shift slightly each frame → visible shimmer at idle.

**Fix:** Dead zone + slow-converge stabilizer on `cached_target_face.kps` and `.bbox`:
- `_NOISE_FLOOR = 0.5` px: deltas below this are discarded (held at previous position).
- `_CONVERGE_CEIL = 2.5` px: deltas 0.5–2.5px drift toward the new position at α=0.3 per frame.
- Deltas above 2.5px (real fast movement): blend at α=0.5–0.95 proportional to speed.

The stabilizer runs on every detection cycle. Because swap, enhancer, and hull mask all consume the
same `cached_target_face` object, a single stabilization point covers all three — no misalignment
between them.

---

### 4. Elliptical paste mask corners producing visible box artifact
**Cause:** The original `_get_soft_alpha` in `face_swapper.py` used an eroded square alpha template.
The straight edges of the square were near-opaque in aligned-face space and warped back as a visible
box on the output face, especially against plain backgrounds.

**Fix:** Replaced with an elliptical template (axes 0.44×size, GaussianBlur 31×31 σ=12). The ellipse
zeros the corners; the heavy blur feathers smoothly into the original frame.

---

### 5. Poisson blend wobble
**Cause:** The old `poisson_blend` path built its seamlessClone mask from `create_face_mask()`
(independently detected landmarks). Those landmarks jitter sub-pixel every frame; seamlessClone is
hypersensitive to mask boundary shifts → wobble even when face is still.

**Fix:** `_apply_poisson_blend()` in `face_swapper.py`. Preferred path derives the blend mask from
the swap's own affine transform (M) + swapped pixels (bgr_fake) via `_create_elliptical_mask()`.
Mask is locked exactly to where the swapped face was placed — no independent jitter source.
Falls back to a bbox-ellipse cached by (center, radius, frame_size) when M is unavailable.

---

### 6. All remaining jitter resolved by CPU load reduction
**Cause:** ROCm/HIP defaults to busy-polling CPU threads to detect GPU completion instead of using
hardware interrupts. With all CPU cores saturated, the `_ProcessingWorker` and `_CaptureWorker`
threads are starved of CPU time, causing irregular frame processing intervals. This produced
inconsistent swap cadence even when stabilization was correctly implemented.

**Fix:** Set the following in `run.py` before any imports when MIGraphX is detected in `sys.argv`:

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

`HSA_ENABLE_INTERRUPT=1` is the decisive fix. The symptom — all cores at 100% in htop — is the
classic HIP spin-poll signature on ROCm. Interrupts reduce idle CPU to near zero, giving worker
threads consistent scheduling and eliminating timing jitter entirely.

---

## Architecture

```
_CaptureWorker  →  capture_queue (maxsize=2, drop-oldest on full)
                         ↓
_ProcessingWorker  (queue always drained → newest frame only)
    detect_one_face_with_landmarks()  →  kps + landmark_2d_106
    dead zone + slow-converge stabilizer  →  cached_target_face
    swap_face()
        paste_back=False  →  bgr_fake + M
        _face_mask_paste_back()  →  landmark convex-hull blend
    enhance_face_inline()  [GPEN, uses same cached_target_face]
                         ↓
               processed_queue (maxsize=2)
                         ↓
                      Display
```

## Key Functions

| Function | File | Purpose |
|---|---|---|
| `detect_one_face_with_landmarks` | `face_analyser.py` | Det + landmark_2d_106, no recognition |
| `_face_mask_paste_back` | `face_swapper.py` | Blend using face-contour convex-hull mask |
| `create_face_mask` | `face_swapper.py` | Feathered convex-hull mask from landmark_2d_106 |
| `_fast_paste_back` | `face_swapper.py` | Fallback: feathered rectangular affine mask |
| `_get_soft_alpha` | `face_swapper.py` | Cached elliptical feathered 128×128 alpha template |
| `_apply_poisson_blend` | `face_swapper.py` | Poisson blend via swap affine, no landmark jitter |
| `_create_elliptical_mask` | `face_swapper.py` | Geometry-based elliptical mask, cached by size |

## Hardware Notes (RX 9070 XT / MIGraphX)

- MIGraphX achieves ~30 face swaps/second on this GPU — matches webcam framerate.
- FP16 is not the cause of any quality issues (confirmed).
- MIGraphX model cache at `~/.cache/migraphx_models/` avoids recompilation on startup.
- `migraphx_fp16_enable=1` is set in the provider config for `get_face_swapper()`.

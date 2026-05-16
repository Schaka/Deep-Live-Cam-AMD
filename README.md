# Deep-Live-Cam (AMD / MiGraphX Fork)

Real-time face swap and video deepfake with a single click and a single image.

This is a fork of [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam), focused exclusively on AMD GPUs with ROCm 7.0+ and the MiGraphX execution provider. The upstream repository has effectively stopped receiving open-source updates. The maintainers chose to take an AGPLv3-licensed project and move meaningful development behind a paid download wall. Whether or not that is strictly illegal under the license is a question for lawyers. That it is a significant breach of open-source community norms is not. This fork exists so AMD users have a working, maintained, and actually open alternative.

---

## Disclaimer

This software is designed for legitimate creative use: animating custom characters, content creation, and similar AI-generated media applications. Built-in checks block inappropriate content (nudity, graphic material, etc.).

- Obtain consent before using a real person's face.
- Label any output clearly as a deepfake when sharing.
- We are not responsible for misuse. Users are responsible for legal and ethical compliance.

---

## Why This Fork

- The original codebase provided no working AMD GPU path beyond basic CPU fallback.
- ONNX Runtime dropped its ROCm execution provider in favor of MiGraphX. The upstream code was not updated to reflect this.
- The upstream project now ships new features exclusively in a paid binary while keeping the public repository stagnant.
- AMD users with modern GPUs (tested on RX 9070 XT) had no working real-time path.

This fork targets ROCm 7.0+ with MiGraphX, using the `onnxruntime-migraphx` wheel from AMD's official repository. The goal is to push as much work as possible onto the GPU so the CPU stays free. All major inference models (face detection, face swapper, face enhancer) run on the GPU via MiGraphX with FP16 where supported.

---

## Provided As-Is

This fork is a personal project. It works on the hardware it was tested on (RX 9070 XT, ROCm 7.x) and is shared in case it helps other AMD users. It is not production software. Some code paths were added during experimentation and have not been fully cleaned up. Some may not be strictly necessary anymore depending on your hardware.

**The minimum viable change to make the official upstream code work with MiGraphX** is just this block at the very top of `run.py`, before any other imports:

```python
import sys, os

# Must be set before ANY import so OMP/BLAS/HIP runtimes read them at init time.
_is_migraphx = any('migraphx' in a.lower() for a in sys.argv)
if _is_migraphx:
    # Read --execution-threads from argv before argparse runs.
    _exec_threads = '1'  # matches suggest_execution_threads() default for MIGraphX
    for _i, _arg in enumerate(sys.argv):
        if _arg == '--execution-threads' and _i + 1 < len(sys.argv):
            _exec_threads = sys.argv[_i + 1]
            break

    # CPU thread pools used by ORT CPU fallback, OpenBLAS, and OpenMP.
    os.environ.setdefault('OMP_NUM_THREADS', _exec_threads)
    os.environ.setdefault('MKL_NUM_THREADS', _exec_threads)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', _exec_threads)
    os.environ.setdefault('GOTO_NUM_THREADS', _exec_threads)
    # Force ROCm/HIP to use interrupt-based GPU completion signalling instead
    # of busy-polling CPU threads -- the primary cause of 100% CPU on all cores.
    os.environ.setdefault('HSA_ENABLE_INTERRUPT', '1')
    # Reduce HIP hardware queues (default can be 8+ per device).
    os.environ.setdefault('GPU_MAX_HW_QUEUES', '1')
```

That alone will get MiGraphX running with the upstream code. You will still see a rectangular boundary around the swapped face on high-contrast backgrounds (the original square paste-back behavior), but it is not always obvious depending on the scene.

The additional changes in this fork (landmark-based face mask blending, sync mode, ORT session thread limits, etc.) were built on top of that foundation to address specific artifacts. Many of them may not be necessary if your GPU is fast enough to process frames at or above webcam framerate without the CPU being hammered by HIP busy-polling. The interrupt mode fix alone removes most of the timing jitter. Everything else is incremental improvement.

Sync mode in particular was an attempt to address a mismatch between the rate at which the model produces swapped frames and the rate at which the webcam delivers new ones. On a GPU that keeps up with the webcam, the queue draining approach (async default) is sufficient and sync mode offers little benefit.

---

## Key Changes in This Fork

### Face Mask Blending (no more rectangle artifacts)

The original code pasted the swapped face back onto the frame using a rectangular affine boundary derived from the 128x128 inswapper crop area. Against high-contrast backgrounds (walls, door frames, etc.) this produced a visible rectangular or quadrilateral edge around the swapped face.

This fork replaces that approach with a convex-hull mask derived from the `landmark_2d_106` facial landmarks. The mask follows the actual face contour with a feathered blend edge, so the swap blends naturally into the background regardless of what is behind the face. The face analyser was updated to use a combined detection + landmark model (`detect_one_face_with_landmarks`) so landmarks are always available during the live loop.

The old rectangular paste-back is kept as a fallback for cases where landmarks are unavailable.

### Sync Mode and Jitter Fixes

In async (default) mode, the capture queue can accumulate frames faster than the GPU processes them. The processing worker would consume a frame that was already 1-2 frames stale by display time. The face-shaped mask was drawn from accurate current landmarks, but the swap was placed using stale keypoints, causing visible misalignment during movement.

Fixes applied:

- **Queue draining in async mode:** after pulling a frame from the capture queue, any remaining queued frames are discarded and only the newest is processed. This bounds display lag to approximately one capture interval (~33ms at 30fps).
- **Sync mode toggle:** a new "Sync mode" option in the UI disables queue draining and processes every frame in order. In sync mode, frames are always fresh and consecutive, so the face moves smoothly without needing stabilization. The stabilizer (dead-zone + convergence blend) is disabled in sync mode because it would cause the swap keypoints to lag behind the accurate landmark positions, creating the very misalignment it was meant to prevent.
- **Stabilizer retained for async mode:** in async mode, the dead-zone stabilizer still runs to smooth out the jumps caused by stale frames.

### CPU Load Reduction (ROCm / HIP Interrupt Mode)

ROCm defaults to busy-polling CPU threads to detect GPU completion instead of using hardware interrupts. With all CPU cores saturated by HIP spin-polling, the worker threads were starved of CPU time, causing irregular frame timing that produced jitter even when queue draining and stabilization were correctly implemented.

Fixes applied when MiGraphX is detected:

- `HSA_ENABLE_INTERRUPT=1`: switches ROCm from spin-polling to interrupt mode. This is the decisive fix. CPU usage drops from near 100% across all cores to near zero at idle.
- `GPU_MAX_HW_QUEUES=1`: one HIP queue per device instead of the default 8+.
- Thread count environment variables (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `GOTO_NUM_THREADS`) set to 1.
- ONNX Runtime `SessionOptions` with `inter_op_num_threads=1` and `intra_op_num_threads=1` on all sessions.
- `cv2.setNumThreads(1)` at the start of the processing worker when MiGraphX is active.

The symptom (all cores at 100%, all appearing as separate `python3 run.py` processes in htop) is the classic HIP spin-poll signature on ROCm. Switching to interrupt mode resolves it completely.

### MiGraphX Provider Configuration

- FP16 inference enabled via `migraphx_fp16_enable=1` in provider config.
- Model cache at `~/.cache/migraphx_models/` to avoid MiGraphX recompilation on every startup.
- The `inswapper_128_fp16.onnx` model is preferred over the FP32 variant.

For more detail on jitter root causes and solutions, see [jitter_handling.md](jitter_handling.md).

---

## Models Required

Place models in the `models/` directory.

| Model | Link |
|---|---|
| `inswapper_128_fp16.onnx` | [Download](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx) |
| `GFPGANv1.4.onnx` | [Download](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx) |

The FP16 inswapper is required. The FP32 variant (`inswapper_128.onnx`) works as a fallback but is slower and not recommended.

---

## Installation (Linux / AMD GPU)

This setup is tested on ROCm 7.0+ with MiGraphX. Python 3.12 is required throughout.

### Recommended: Distrobox

Using [distrobox](https://github.com/89luca89/distrobox) avoids polluting your host system and gives you a clean ROCm environment with all GPU libraries pre-installed.

**1. Create the container**

Use the official AMD ONNX Runtime image. Pick a tag matching your ROCm version:

```
rocm/onnxruntime:rocm7.2.3_ub24.04_ort1.23_torch2.10.0
```

Available tags: https://hub.docker.com/r/rocm/onnxruntime

```bash
distrobox create --name dlc-amd --image rocm/onnxruntime:rocm7.2.3_ub24.04_ort1.23_torch2.10.0
distrobox enter dlc-amd
```

**2. Install system dependencies**

```bash
sudo apt install git ffmpeg python3-tk
```

**3. Clone the repo**

```bash
git clone https://github.com/Schaka/Deep-Live-Cam.git
cd Deep-Live-Cam
```

**4. Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

**5. Install Python dependencies**

Remove the generic onnxruntime packages and install requirements:

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install -r requirements.txt
```

**6. Install the MiGraphX ONNX Runtime wheel**

Grab the correct wheel from AMD's repository. The wheel must match your Python version (3.12) and ROCm version.

Browse available wheels: https://repo.radeon.com/rocm/manylinux/

Example for ROCm 7.2.3:

```bash
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Adjust the URL to match your ROCm version and the available wheel filename.

**7. Verify GPU provider is available**

```bash
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

`MIGraphXExecutionProvider` must appear in the output.

**8. Download models**

```bash
mkdir -p models
wget -O models/inswapper_128_fp16.onnx \
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx
wget -O models/GFPGANv1.4.onnx \
  https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx
```

**9. Run**

```bash
python3 run.py --execution-provider migraphx
```

On first run, MiGraphX will compile the models and cache them in `~/.cache/migraphx_models/`. Subsequent starts are significantly faster.

---

## Usage

**Webcam / Live Mode**

1. Run `python3 run.py --execution-provider migraphx`
2. Select a source face image
3. Click "Live"
4. Wait for the preview (first run takes longer due to MiGraphX model compilation)
5. Use OBS or any screen capture tool to stream

**Image / Video Mode**

1. Run `python3 run.py --execution-provider migraphx`
2. Choose a source face image and a target image or video
3. Click "Start"
4. Output is saved in a directory named after the target file

---

## UI Options (MiGraphX / AMD Specific)

| Option | Description |
|---|---|
| Sync mode | Process every frame in order. Reduces jitter during fast movement at the cost of potential FPS reduction if the GPU cannot keep up with the webcam framerate. |
| Show FPS | Display swap rate counter on the live preview. |

---

## Performance Notes

- RX 9070 XT achieves approximately 30 face swaps per second with MiGraphX, matching a 30fps webcam.
- The goal is zero meaningful CPU usage during inference. All detection, swapping, and enhancement runs on the GPU.
- `HSA_ENABLE_INTERRUPT=1` is critical. Without it, ROCm busy-polls CPU cores and causes timing jitter that no amount of queue or stabilizer tuning can fix.
- MiGraphX model cache at `~/.cache/migraphx_models/` is created automatically.
- FP16 is enabled by default and does not cause quality issues.

---

## Command Line Arguments

```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select a source image
  -t TARGET_PATH, --target TARGET_PATH                     select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  frame processors (choices: face_swapper, face_enhancer, ...)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --map-faces                                              map source target faces
  --mouth-mask                                             mask the mouth region
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  --live-mirror                                            mirror the live camera display
  --live-resizable                                         make the live camera frame resizable
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu,migraphx,cuda,...}             available execution provider
  --execution-threads EXECUTION_THREADS                    number of execution threads
  -v, --version                                            show program's version number and exit
```

---

## Known Limitations

- **Face enhancer:** the GFPGAN and GPEN enhancer options exist in the UI but are not well tested with MiGraphX. The original codebase provides no documentation on how these models are expected to behave under non-CUDA providers. They may produce incorrect results or fail silently. Disable them if the output looks wrong.
- **Many faces / Map faces:** these modes are inherited from upstream and have not been specifically tested with the MiGraphX code paths in this fork.
- This fork targets live webcam mode first. Image/video batch processing should work but is secondary.

---

## Credits

- [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam): original project this fork is based on
- [s0md3v/roop](https://github.com/s0md3v/roop): base codebase
- [ffmpeg](https://ffmpeg.org/): video operations
- [deepinsight/insightface](https://github.com/deepinsight/insightface): face detection and swapper models (non-commercial research use only per their license)

# How to Run on Linux with AMD (ROCm 7.0+ / MiGraphX)

## Requirements

- AMD GPU with ROCm 7.0+ support
- Python 3.12
- ROCm drivers installed on the host (or inside the container)

## Recommended: Distrobox

Distrobox creates an isolated container based on an official AMD image that already ships ROCm, ONNX Runtime, and PyTorch. This avoids driver conflicts and keeps your host clean.

**Create the container:**

```bash
distrobox create --name dlc-amd --image rocm/onnxruntime:rocm7.2.3_ub24.04_ort1.23_torch2.10.0
distrobox enter dlc-amd
```

Pick a tag from https://hub.docker.com/r/rocm/onnxruntime that matches your installed ROCm version. The image ships Ubuntu 24.04 with Python 3.12.

## Setup Inside the Container

```bash
# System dependencies
sudo apt install git ffmpeg python3-tk

# Clone the repo
git clone https://github.com/Schaka/Deep-Live-Cam.git
cd Deep-Live-Cam

# Create and activate venv (use the system Python 3.12)
python3 -m venv venv
source venv/bin/activate

# Remove any generic onnxruntime packages
pip uninstall onnxruntime onnxruntime-gpu -y

# Install project dependencies
pip install -r requirements.txt

# Install the MiGraphX ONNX Runtime wheel
# Browse https://repo.radeon.com/rocm/manylinux/ for your ROCm version and Python 3.12 wheel
# Example for ROCm 7.2.3:
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/onnxruntime_migraphx-1.23.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

# Verify the provider is available
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# MIGraphXExecutionProvider must appear in the output
```

## Download Models

```bash
mkdir -p models

# Required: FP16 inswapper (preferred over FP32)
wget -O models/inswapper_128_fp16.onnx \
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx

# Optional: face enhancer
wget -O models/GFPGANv1.4.onnx \
  https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx
```

## Run

```bash
python3 run.py --execution-provider migraphx
```

On first launch, MiGraphX compiles and caches the models in `~/.cache/migraphx_models/`. This takes a minute or two. Subsequent launches are fast.

## Notes

- This fork does NOT use a ROCm-specific onnxruntime package. ONNX Runtime dropped its ROCm execution provider. The correct path is now the MiGraphX execution provider via the `onnxruntime-migraphx` wheel from AMD's repository.
- Use Python 3.12. The AMD wheels are built for 3.12.
- `HSA_ENABLE_INTERRUPT=1` is set automatically by `run.py` when MiGraphX is detected. This switches ROCm from CPU busy-polling to interrupt mode, which is critical for stable performance and low CPU usage.
- All inference (face detection, swapper, enhancer) runs on the GPU via MiGraphX. CPU usage should be near zero during normal operation.

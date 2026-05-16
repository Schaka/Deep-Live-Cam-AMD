#!/usr/bin/env python3

import os
import sys

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
    # of busy-polling CPU threads — the primary cause of 100% CPU on all cores.
    os.environ.setdefault('HSA_ENABLE_INTERRUPT', '1')
    # Reduce HIP hardware queues (default can be 8+ per device).
    os.environ.setdefault('GPU_MAX_HW_QUEUES', '1')

# Add the project root to PATH so bundled ffmpeg/ffprobe are found
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# On Windows, register NVIDIA CUDA DLL directories so onnxruntime-gpu can
# find cuDNN/cublas. Python 3.8+ ignores PATH for extension-module native deps —
# os.add_dll_directory() is required. Also keep PATH for child processes/ffmpeg.
if sys.platform == "win32":
    _site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    _venv_site_packages = os.path.join(project_root, "venv", "Lib", "site-packages")
    for _sp in (_site_packages, _venv_site_packages):
        _candidate_dirs = []
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            _candidate_dirs.append(_torch_lib)
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    _candidate_dirs.append(_bin_dir)
        for _d in _candidate_dirs:
            os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass

from modules import platform_info
platform_info.print_banner()

from modules import core

if __name__ == '__main__':
    core.run()

"""
Convert GFPGANv1.4.pth to gfpgan-1024.onnx for use with the ONNX face enhancer.

Requirements (install once, not needed at runtime):
    pip install torch torchvision gfpgan basicsr facexlib

Usage:
    python convert_gfpgan_to_onnx.py
"""

import os
import sys
import types
import torch
import numpy as np

# basicsr/gfpgan reference a module removed in torchvision >= 0.16
import torchvision.transforms.functional as _tvf
_mock = types.ModuleType("torchvision.transforms.functional_tensor")
_mock.rgb_to_grayscale = _tvf.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _mock

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PTH_PATH = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")
ONNX_PATH = os.path.join(MODELS_DIR, "gfpgan-1024.onnx")

INPUT_SIZE = 512  # GFPGANv1.4 operates at 512x512


def load_gfpgan_net(pth_path: str):
    """Load only the GFPGANv1Clean network (no full GFPGANer wrapper needed)."""
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

    net = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )

    checkpoint = torch.load(pth_path, map_location="cpu")
    # The .pth may store the net under a 'params_ema' or 'params' key
    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def export(net, onnx_path: str, input_size: int):
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        net,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed size — enables more constant folding
        dynamo=False,  # use legacy exporter; new dynamo exporter needs onnxscript
    )
    print(f"Saved: {onnx_path}")

    # Simplify graph to collapse reshape chains that MiGraphX simplify_reshapes
    # can't handle on StyleGAN2-based models (causes std::bad_alloc).
    try:
        import onnxsim
        import onnx
        print("Running onnxsim to simplify graph for MiGraphX compatibility...")
        model = onnx.load(onnx_path)
        simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(simplified, onnx_path)
            print("Graph simplified successfully.")
        else:
            print("onnxsim could not fully simplify — model saved as-is.")
    except ImportError:
        print("onnxsim not installed — skipping simplification.")
        print("If MiGraphX fails to load the model, run:")
        print("  pip install onnxsim && python -m onnxsim models/gfpgan-1024.onnx models/gfpgan-1024.onnx")


def main():
    if not os.path.exists(PTH_PATH):
        print(f"ERROR: {PTH_PATH} not found.")
        sys.exit(1)

    if os.path.exists(ONNX_PATH):
        answer = input(f"{ONNX_PATH} already exists. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"Loading {PTH_PATH} ...")
    net = load_gfpgan_net(PTH_PATH)

    print(f"Exporting to {ONNX_PATH} (input size {INPUT_SIZE}x{INPUT_SIZE}) ...")
    export(net, ONNX_PATH, INPUT_SIZE)
    print("Done.")


if __name__ == "__main__":
    main()

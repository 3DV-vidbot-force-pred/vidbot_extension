"""Estimate metric depth from RGB frames using Metric3D or Depth Anything V3.

Usage:
    python scripts/estimate_depth.py --dataset my_video --model metric3d
    python scripts/estimate_depth.py --dataset my_video --model dav3
    python scripts/estimate_depth.py --dataset my_video --model dav3 --frames 0 10 20

Reads  : datasets/<name>/color/*.png  (or .jpg)
Writes : datasets/<name>/depth_m3d/*.png   (Metric3D)
         datasets/<name>/depth_dav3/*.png  (Depth Anything V3)

Depth maps are saved as uint16 PNG in millimetres (matching VidBot convention).
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vidbot_utils.device import get_device


# ---------------------------------------------------------------------------
# Metric3D
# ---------------------------------------------------------------------------

def load_metric3d(variant="ViT-Small", device=None):
    """Load a Metric3D model from the local submodule."""
    if device is None:
        device = get_device()
    metric3d_root = os.path.join(os.path.dirname(__file__), "..", "third_party", "Metric3D")
    sys.path.insert(0, metric3d_root)

    try:
        from mmcv.utils import Config
    except ImportError:
        from mmengine import Config

    from mono.model.monodepth_model import get_configured_monodepth_model

    MODEL_TYPE = {
        "ViT-Small": {
            "cfg": os.path.join(metric3d_root, "mono/configs/HourglassDecoder/vit.raft5.small.py"),
            "ckpt": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth",
            "input_size": (616, 1064),
        },
        "ViT-Large": {
            "cfg": os.path.join(metric3d_root, "mono/configs/HourglassDecoder/vit.raft5.large.py"),
            "ckpt": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth",
            "input_size": (616, 1064),
        },
        "ConvNeXt-Large": {
            "cfg": os.path.join(metric3d_root, "mono/configs/HourglassDecoder/convlarge.0.3_150.py"),
            "ckpt": "https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth",
            "input_size": (544, 1216),
        },
    }

    info = MODEL_TYPE[variant]
    cfg = Config.fromfile(info["cfg"])
    model = get_configured_monodepth_model(cfg)
    state = torch.hub.load_state_dict_from_url(info["ckpt"], map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, info["input_size"]


def predict_metric3d(model, rgb_bgr, intrinsic, input_size, device):
    """Run Metric3D on a single BGR image. Returns depth in metres (H, W)."""
    rgb = rgb_bgr[:, :, ::-1].copy()  # BGR -> RGB
    h, w = rgb.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb_resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # Scale intrinsic
    fx, fy, cx, cy = intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale

    # Pad to input_size
    padding_val = [123.675, 116.28, 103.53]
    rh, rw = rgb_resized.shape[:2]
    pad_h, pad_w = input_size[0] - rh, input_size[1] - rw
    pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
    rgb_padded = cv2.copyMakeBorder(
        rgb_resized, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
        cv2.BORDER_CONSTANT, value=padding_val,
    )

    # Normalise
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]
    tensor = torch.from_numpy(rgb_padded.transpose(2, 0, 1)).float()
    tensor = (tensor - mean) / std
    tensor = tensor[None].to(device)

    with torch.no_grad():
        pred_depth, _, _ = model.inference({"input": tensor})

    pred_depth = pred_depth.squeeze()
    # Un-pad
    pred_depth = pred_depth[pad_h_half:pred_depth.shape[0] - (pad_h - pad_h_half),
                            pad_w_half:pred_depth.shape[1] - (pad_w - pad_w_half)]
    # Resize back to original
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None], size=(h, w), mode="bilinear", align_corners=False,
    ).squeeze()

    # De-canonical transform (Metric3D uses canonical focal = 1000)
    canonical_to_real = fx / 1000.0
    pred_depth = pred_depth * canonical_to_real
    pred_depth = torch.clamp(pred_depth, 0, 10)

    return pred_depth.cpu().numpy()


# ---------------------------------------------------------------------------
# Depth Anything V3
# ---------------------------------------------------------------------------

def load_dav3(model_name="da3-large", device=None):
    """Load a Depth Anything V3 model."""
    if device is None:
        device = get_device()
    dav3_root = os.path.join(os.path.dirname(__file__), "..", "third_party", "DepthAnythingV3")
    if dav3_root not in sys.path:
        sys.path.insert(0, os.path.join(dav3_root, "src"))

    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}")
    model.to(device).eval()
    return model


def predict_dav3(model, rgb_bgr, intrinsic=None):
    """Run DAv3 on a single BGR image. Returns depth in metres (H, W)."""
    from PIL import Image
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    intrinsics_np = None
    if intrinsic is not None:
        fx, fy, cx, cy = intrinsic
        intrinsics_np = np.array([
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        ], dtype=np.float32)

    prediction = model.inference(
        [pil_img],
        intrinsics=torch.from_numpy(intrinsics_np) if intrinsics_np is not None else None,
        export_format="mini_npz",
        export_dir=None,
    )
    depth = prediction.depth[0]  # (H, W) numpy
    return depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_intrinsic(dataset_path):
    """Load camera intrinsic as [fx, fy, cx, cy]."""
    import json
    intr_path = os.path.join(dataset_path, "camera_intrinsic.json")
    if not os.path.exists(intr_path):
        return None
    with open(intr_path) as f:
        info = json.load(f)
    mat = np.array(info["intrinsic_matrix"]).reshape(3, 3).T.astype(np.float32)
    return [mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2]]  # fx, fy, cx, cy


def main():
    parser = argparse.ArgumentParser(description="Estimate metric depth from RGB frames")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset folder name inside datasets/")
    parser.add_argument("-m", "--model", choices=["metric3d", "dav3"], default="dav3",
                        help="Depth model to use (default: dav3)")
    parser.add_argument("--variant", default=None,
                        help="Model variant (Metric3D: ViT-Small/ViT-Large/ConvNeXt-Large, DAv3: da3-large/da3-giant)")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="Specific frame indices to process (default: all)")
    parser.add_argument("--dataset_dir", default="./datasets", help="Root dataset directory")
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir, args.dataset)
    color_dir = os.path.join(dataset_path, "color")
    assert os.path.isdir(color_dir), f"Color directory not found: {color_dir}"

    # Determine output directory
    if args.model == "metric3d":
        out_dir = os.path.join(dataset_path, "depth_m3d")
    else:
        out_dir = os.path.join(dataset_path, "depth_dav3")
    os.makedirs(out_dir, exist_ok=True)

    # Find frames
    color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")) +
                         glob.glob(os.path.join(color_dir, "*.jpg")))
    if args.frames is not None:
        color_files = [
            f for f in color_files
            if int(os.path.splitext(os.path.basename(f))[0]) in args.frames
        ]
    print(f"Processing {len(color_files)} frames with {args.model}")

    device = get_device()
    intrinsic = load_intrinsic(dataset_path)
    if intrinsic is not None:
        print(f"Camera intrinsic: fx={intrinsic[0]:.1f} fy={intrinsic[1]:.1f} cx={intrinsic[2]:.1f} cy={intrinsic[3]:.1f}")
    else:
        print("WARNING: No camera_intrinsic.json found — depth scale may be inaccurate")

    # Load model
    if args.model == "metric3d":
        variant = args.variant or "ViT-Small"
        print(f"Loading Metric3D ({variant})...")
        model, input_size = load_metric3d(variant, device)
    else:
        variant = args.variant or "da3-large"
        print(f"Loading Depth Anything V3 ({variant})...")
        model = load_dav3(variant, device)

    # Process frames
    for color_path in color_files:
        fname = os.path.basename(color_path)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(out_dir, f"{stem}.png")

        if os.path.exists(out_path):
            print(f"  {stem} — already exists, skipping")
            continue

        bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  {stem} — failed to read, skipping")
            continue

        if args.model == "metric3d":
            assert intrinsic is not None, "Metric3D requires camera intrinsics"
            depth_m = predict_metric3d(model, bgr, intrinsic, input_size, device)
        else:
            depth_m = predict_dav3(model, bgr, intrinsic)

        # Convert metres -> millimetres, save as uint16
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(out_path, depth_mm)
        print(f"  {stem} — depth range: {depth_m.min():.3f}m .. {depth_m.max():.3f}m")

    print(f"\nDone. Depth maps saved to {out_dir}")


if __name__ == "__main__":
    main()

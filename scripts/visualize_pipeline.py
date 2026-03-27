"""Visualize VidBot pipeline intermediate results.

Creates a multi-panel figure showing:
  1. RGB input frame
  2. Estimated depth map (colorized)
  3. 3D point cloud with camera coordinate frame
  4. Depth overlay on RGB
  5. Intrinsic verification (reprojection grid)
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def load_intrinsic(dataset_path):
    """Load camera intrinsic as 3x3 matrix."""
    with open(os.path.join(dataset_path, "camera_intrinsic.json")) as f:
        info = json.load(f)
    mat = np.array(info["intrinsic_matrix"]).reshape(3, 3).T.astype(np.float32)
    return mat


def backproject(depth, intr, mask=None, subsample=8):
    """Backproject depth to 3D points."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    if mask is None:
        mask = depth > 0.01
    valid = mask & (depth > 0.01) & (depth < 5.0)
    u, v, z = u[valid], v[valid], depth[valid]
    x = (u - intr[0, 2]) * z / intr[0, 0]
    y = (v - intr[1, 2]) * z / intr[1, 1]
    points = np.stack([x, y, z], axis=-1)
    # Subsample for visualization
    if subsample > 1 and len(points) > 5000:
        idx = np.random.choice(len(points), len(points) // subsample, replace=False)
        return points[idx], valid
    return points, valid


def visualize_frame(dataset_path, frame_id, depth_dir="depth_m3d", out_path=None):
    """Create comprehensive visualization for a single frame."""
    color_path = os.path.join(dataset_path, "color", f"{frame_id:06d}.png")
    if not os.path.exists(color_path):
        color_path = color_path.replace(".png", ".jpg")
    depth_path = os.path.join(dataset_path, depth_dir, f"{frame_id:06d}.png")

    color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[depth_m > 5.0] = 0

    intr = load_intrinsic(dataset_path)
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    # --- Create figure ---
    fig = plt.figure(figsize=(24, 16), facecolor="white")
    fig.suptitle(
        f"VidBot Pipeline Visualization — Frame {frame_id}\n"
        f"Intrinsics: fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # 1. RGB Input
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(color_rgb)
    ax1.set_title("1. RGB Input", fontsize=14, fontweight="bold")
    ax1.axis("off")

    # 2. Depth Map (colorized)
    ax2 = fig.add_subplot(2, 3, 2)
    valid_mask = depth_m > 0.01
    vmin = np.percentile(depth_m[valid_mask], 2) if valid_mask.any() else 0
    vmax = np.percentile(depth_m[valid_mask], 98) if valid_mask.any() else 3
    depth_vis = np.where(valid_mask, depth_m, np.nan)
    im = ax2.imshow(depth_vis, cmap="turbo", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax2, label="Depth (m)", shrink=0.8)
    ax2.set_title(f"2. Metric3D Depth ({depth_dir})", fontsize=14, fontweight="bold")
    ax2.axis("off")

    # 3. Depth overlay on RGB
    ax3 = fig.add_subplot(2, 3, 3)
    depth_color = plt.cm.turbo((depth_m - vmin) / (vmax - vmin + 1e-8))[:, :, :3]
    depth_color = (depth_color * 255).astype(np.uint8)
    overlay = color_rgb.copy()
    overlay[valid_mask] = cv2.addWeighted(
        color_rgb[valid_mask], 0.4, depth_color[valid_mask], 0.6, 0
    )
    ax3.imshow(overlay)
    ax3.set_title("3. Depth Overlay on RGB", fontsize=14, fontweight="bold")
    ax3.axis("off")

    # 4. Depth histogram
    ax4 = fig.add_subplot(2, 3, 4)
    valid_depths = depth_m[valid_mask]
    ax4.hist(valid_depths, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax4.set_xlabel("Depth (m)", fontsize=12)
    ax4.set_ylabel("Pixel count", fontsize=12)
    ax4.set_title("4. Depth Distribution", fontsize=14, fontweight="bold")
    ax4.axvline(np.median(valid_depths), color="red", linestyle="--", label=f"median={np.median(valid_depths):.2f}m")
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # 5. 3D Point Cloud with Camera Frame (top view)
    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    points, _ = backproject(depth_m, intr, subsample=16)
    # Get colors for points
    h, w = depth_m.shape
    u_all, v_all = np.meshgrid(np.arange(w), np.arange(h))
    mask_flat = (depth_m > 0.01) & (depth_m < 5.0)
    u_valid = u_all[mask_flat]
    v_valid = v_all[mask_flat]
    colors_valid = color_rgb[mask_flat] / 255.0
    # Subsample colors matching points
    np.random.seed(42)
    n_pts = len(points)
    if len(colors_valid) > n_pts:
        idx = np.random.choice(len(colors_valid), n_pts, replace=False)
        colors_vis = colors_valid[idx]
    else:
        colors_vis = colors_valid[:n_pts]

    ax5.scatter(points[:, 0], points[:, 2], -points[:, 1],
                c=colors_vis, s=0.3, alpha=0.6)

    # Camera coordinate frame (at origin)
    axis_len = 0.15
    # X axis (red)
    arrow_x = Arrow3D([0, axis_len], [0, 0], [0, 0],
                       mutation_scale=15, lw=3, arrowstyle="-|>", color="red")
    ax5.add_artist(arrow_x)
    # Y axis (green) — pointing down in camera frame, so -Y for display
    arrow_y = Arrow3D([0, 0], [0, 0], [0, axis_len],
                       mutation_scale=15, lw=3, arrowstyle="-|>", color="green")
    ax5.add_artist(arrow_y)
    # Z axis (blue) — optical axis
    arrow_z = Arrow3D([0, 0], [0, axis_len], [0, 0],
                       mutation_scale=15, lw=3, arrowstyle="-|>", color="blue")
    ax5.add_artist(arrow_z)

    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Z (m)")
    ax5.set_zlabel("-Y (m)")
    ax5.set_title("5. 3D Point Cloud + Camera Frame", fontsize=14, fontweight="bold")
    ax5.view_init(elev=-60, azim=-90)

    # 6. Intrinsic verification — show projected grid
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(color_rgb, alpha=0.5)
    # Draw principal point
    ax6.plot(cx, cy, "r+", markersize=20, markeredgewidth=3, label=f"Principal pt ({cx:.0f}, {cy:.0f})")
    # Draw image center
    img_cx, img_cy = w / 2, h / 2
    ax6.plot(img_cx, img_cy, "bx", markersize=15, markeredgewidth=2, label=f"Image center ({img_cx:.0f}, {img_cy:.0f})")
    # Draw FoV lines from principal point
    for angle_h in [-56.5, -30, 0, 30, 56.5]:  # ~113 deg horizontal
        rad = np.radians(angle_h)
        u_edge = cx + fx * np.tan(rad)
        ax6.axvline(u_edge, color="orange", alpha=0.3, linewidth=0.8)
    for angle_v in [-40, -20, 0, 20, 40]:
        rad = np.radians(angle_v)
        v_edge = cy + fy * np.tan(rad)
        ax6.axhline(v_edge, color="cyan", alpha=0.3, linewidth=0.8)
    ax6.set_title("6. Intrinsic Verification (FoV Grid)", fontsize=14, fontweight="bold")
    ax6.legend(fontsize=9, loc="lower right")
    ax6.set_xlim(0, w)
    ax6.set_ylim(h, 0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is None:
        out_path = os.path.join(dataset_path, f"viz_frame_{frame_id:06d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved visualization: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline intermediates")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-f", "--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--depth_dir", default="depth_m3d")
    parser.add_argument("--dataset_dir", default="./datasets")
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir, args.dataset)
    for frame_id in args.frames:
        visualize_frame(dataset_path, frame_id, args.depth_dir)


if __name__ == "__main__":
    main()

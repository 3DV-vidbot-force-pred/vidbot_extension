#!/usr/bin/env python3
"""Run the full VidBot pipeline on every clip in desktop_dataset_import_summary.json.

Steps per clip (each skipped if output already exists):
  1. estimate_depth     — Metric3D depth estimation
  2. visualize_pipeline — RGB / depth / point-cloud sanity figure
  3. infer_affordance   — VidBot affordance inference (scene_meta + predictions)
  4. visualize_results  — heatmaps + trajectory figures
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def write_status(path: Path, status: list[dict]) -> None:
    with path.open("w") as f:
        json.dump(status, f, indent=2)


def main() -> None:
    root         = Path(__file__).resolve().parents[1]
    summary_path = root / "datasets" / "desktop_dataset_import_summary.json"
    status_path  = root / "datasets" / "desktop_dataset_run_status.json"

    with summary_path.open("r") as f:
        entries = json.load(f)

    python  = sys.executable
    env_std = os.environ.copy()
    env_cpu = {**env_std, "VIDBOT_DEVICE": "cpu"}

    # Load existing status so re-runs don't reset completed entries
    if status_path.exists():
        with status_path.open("r") as f:
            existing = {s["video"]: s for s in json.load(f)}
    else:
        existing = {}

    status = []

    for entry in entries:
        video       = entry["video"]
        instruction = entry["instruction"]
        obj         = entry["object"]
        clip_dir    = root / "datasets" / video

        item = existing.get(video, {
            "video":              video,
            "instruction":        instruction,
            "object":             obj,
            "depth":              "pending",
            "viz_pipeline":       "pending",
            "vidbot":             "pending",
            "viz_results":        "pending",
        })

        print(f"\n{'='*60}\n{video}  \"{instruction}\"")

        # ── 1. Depth estimation ───────────────────────────────────────────────
        depth_path = clip_dir / "depth_m3d" / "000000.png"
        if depth_path.exists():
            print(f"  [depth] already exists — skipping")
            item["depth"] = "ok"
        else:
            try:
                run(
                    [python, "scripts/estimate_depth.py",
                     "--dataset", video, "--model", "metric3d"],
                    cwd=root, env=env_std,
                )
                item["depth"] = "ok"
            except subprocess.CalledProcessError as exc:
                item["depth"]        = f"failed ({exc.returncode})"
                item["viz_pipeline"] = "skipped"
                item["vidbot"]       = "skipped"
                item["viz_results"]  = "skipped"
                status.append(item)
                write_status(status_path, status)
                continue

        # ── 2. Pipeline visualisation (RGB / depth / point cloud) ─────────────
        viz_pipeline_path = clip_dir / "viz_frame_000000.png"
        if viz_pipeline_path.exists():
            print(f"  [viz_pipeline] already exists — skipping")
            item["viz_pipeline"] = "ok"
        else:
            try:
                run(
                    [python, "scripts/visualize_pipeline.py",
                     "--dataset", video, "--frames", "0"],
                    cwd=root, env=env_std,
                )
                item["viz_pipeline"] = "ok"
            except subprocess.CalledProcessError as exc:
                item["viz_pipeline"] = f"failed ({exc.returncode})"
                # non-fatal — continue to inference

        # ── 3. VidBot affordance inference ────────────────────────────────────
        scene_meta_dir = clip_dir / "scene_meta"
        prediction_dir = clip_dir / "prediction"
        inference_done = (
            scene_meta_dir.exists()
            and prediction_dir.exists()
            and any(prediction_dir.glob("*.npz"))
        )
        if inference_done:
            print(f"  [vidbot] predictions already exist — skipping")
            item["vidbot"] = "ok"
        else:
            try:
                run(
                    [python, "demos/infer_affordance.py",
                     "--dataset", video,
                     "--frame", "0",
                     "--instruction", instruction,
                     "--object", obj,
                     "--depth_model", "auto"],
                    cwd=root, env=env_cpu,
                )
                item["vidbot"] = "ok"
            except subprocess.CalledProcessError as exc:
                item["vidbot"]      = f"failed ({exc.returncode})"
                item["viz_results"] = "skipped"
                status.append(item)
                write_status(status_path, status)
                continue

        # ── 4. Results visualisation (heatmaps + trajectories) ────────────────
        viz_results_dir  = clip_dir / "visualizations"
        viz_results_done = (
            viz_results_dir.exists()
            and any(viz_results_dir.glob("predictions_*.png"))
        )
        if viz_results_done:
            print(f"  [viz_results] already exists — skipping")
            item["viz_results"] = "ok"
        else:
            try:
                run(
                    [python, "scripts/visualize_results.py",
                     "--dataset", video, "--frame", "0"],
                    cwd=root, env=env_std,
                )
                item["viz_results"] = "ok"
            except subprocess.CalledProcessError as exc:
                item["viz_results"] = f"failed ({exc.returncode})"

        status.append(item)
        write_status(status_path, status)

    write_status(status_path, status)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'video':<12} {'depth':<10} {'viz_pipe':<10} {'vidbot':<10} {'viz_res':<10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for s in status:
        print(f"{s['video']:<12} "
              f"{s.get('depth','?'):<10} "
              f"{s.get('viz_pipeline','?'):<10} "
              f"{s.get('vidbot','?'):<10} "
              f"{s.get('viz_results','?'):<10}")
    print(f"\nStatus written to {status_path}")


if __name__ == "__main__":
    main()

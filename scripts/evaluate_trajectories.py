#!/usr/bin/env python3
"""
evaluate_trajectories.py
========================
Quantitative comparison of VidBot vs 3D Diffuser Actor (DA) trajectories.

Coordinate systems
------------------
Both models output trajectories in **camera-frame metric space** (metres):
  X = right, Y = down, Z = forward (depth).

VidBot  pred_trajectories : (1, 40, 81, 3)  — 40 samples, 81 timesteps, XYZ
DA      pred_trajectories : (1, 20, 81, 3)  — 20 samples, 81 timesteps, XYZ
                                              (spline-resampled from DA's 25 → 81)

Important design difference
---------------------------
VidBot *anchors* every sample's t=0 to the predicted contact point, so all 40
starts are within ~1 mm of each other.  This is intentional — VidBot uses a
contact heatmap to fix the start then diffuses the rest of the path.

DA has no such anchoring; samples are drawn from Gaussian noise and vary freely
from t=0 onwards.

Consequence for diversity
-------------------------
Measuring diversity only at t=0 gives ~0.5 mm for VidBot (trivially zero) and
~12 cm for DA — a misleading 200× gap.  Instead we measure:

  diversity  =  mean over all timesteps T of the mean pairwise L2 across samples

This captures how spread the whole trajectory *bundle* is, which is meaningful
for both models.

Metrics
-------
Tier 1 — single-curve / endpoint agreement:
  start_l2_m        L2(VidBot contact, DA mean start)
  end_l2_m          L2(VidBot goal,    DA mean end)
  mean_traj_mse     mean per-timestep MSE between the two mean trajectories
  arc_length_vidbot arc length of VidBot best trajectory (metres)
  arc_length_da     arc length of DA mean trajectory (metres)
  arc_length_ratio  arc_length_da / arc_length_vidbot

Tier 2 — distribution / spread:
  diversity_vidbot  mean-over-T mean-pairwise-L2 across VidBot samples
  diversity_da      same for DA
  chamfer_end_m     symmetric Chamfer distance between endpoint clouds
                    (all N_vb endpoint positions vs all N_da endpoint positions)
  dtw_distance      DTW between the two mean trajectories

Outputs
-------
  datasets/eval_trajectories.csv    one row per (dataset, object)
  datasets/eval_plots/              four PNG figures stratified by force level
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Metric helpers ─────────────────────────────────────────────────────────────

def arc_length(traj: np.ndarray) -> float:
    """
    Total arc length of a (T, 3) trajectory in metres.

    Computed as the sum of Euclidean distances between consecutive waypoints.
    """
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())


def trajectory_diversity(trajs: np.ndarray) -> float:
    """
    Mean spread of a bundle of trajectories across all timesteps.

    For each timestep t, computes the mean pairwise L2 between the N sample
    positions, then averages that value over all T timesteps.

    Parameters
    ----------
    trajs : (N, T, 3)  array of N trajectory samples

    Returns
    -------
    float : metres
    """
    N, T, _ = trajs.shape
    if N < 2:
        return 0.0

    per_t = []
    for t in range(T):
        pts   = trajs[:, t, :]                              # (N, 3)
        diffs = pts[:, None, :] - pts[None, :, :]          # (N, N, 3)
        dists = np.linalg.norm(diffs, axis=-1)             # (N, N)
        # upper triangle only (avoid double-counting)
        mean_pw = dists[np.triu_indices(N, k=1)].mean()
        per_t.append(mean_pw)

    return float(np.mean(per_t))


def chamfer_endpoints(vb_trajs: np.ndarray, da_trajs: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between the two *endpoint* clouds.

    Uses the final timestep of each trajectory as a point, giving two sets:
      P = {vb_traj_i[-1] | i = 0..N_vb}   shape (N_vb, 3)
      Q = {da_traj_j[-1] | j = 0..N_da}   shape (N_da, 3)

    Chamfer(P, Q) = 0.5 * (mean_P min_Q d(p,q) + mean_Q min_P d(p,q))

    Parameters
    ----------
    vb_trajs : (N_vb, T, 3)
    da_trajs : (N_da, T, 3)

    Returns
    -------
    float : metres
    """
    P = vb_trajs[:, -1, :]   # (N_vb, 3)
    Q = da_trajs[:, -1, :]   # (N_da, 3)

    # pairwise distance matrix (N_vb, N_da)
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)

    p_to_q = D.min(axis=1).mean()   # each VidBot endpoint → nearest DA endpoint
    q_to_p = D.min(axis=0).mean()   # each DA endpoint     → nearest VidBot endpoint
    return float((p_to_q + q_to_p) / 2.0)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two (T, D) trajectories.

    Uses standard O(T_a × T_b) DP with Euclidean local cost.
    Both inputs are expected to be mean trajectories (one curve each).

    Parameters
    ----------
    a, b : (T, 3) mean trajectory arrays

    Returns
    -------
    float : cumulative DTW cost (metres)
    """
    T_a, T_b = len(a), len(b)
    cost = np.full((T_a, T_b), np.inf, dtype=np.float64)

    cost[0, 0] = float(np.linalg.norm(a[0] - b[0]))
    for i in range(1, T_a):
        cost[i, 0] = cost[i-1, 0] + float(np.linalg.norm(a[i] - b[0]))
    for j in range(1, T_b):
        cost[0, j] = cost[0, j-1] + float(np.linalg.norm(a[0] - b[j]))
    for i in range(1, T_a):
        for j in range(1, T_b):
            cost[i, j] = float(np.linalg.norm(a[i] - b[j])) + min(
                cost[i-1, j], cost[i, j-1], cost[i-1, j-1]
            )

    return float(cost[-1, -1])


# ── Per-object metric computation ─────────────────────────────────────────────

def compute_metrics(
    vb_trajs:    np.ndarray,   # (N_vb, 81, 3)  VidBot samples in camera-frame metres
    da_trajs:    np.ndarray,   # (N_da, 81, 3)  DA samples in camera-frame metres
    vb_best_idx: int,          # index of VidBot's best sample (lowest guidance loss)
) -> dict[str, float]:
    """
    Compute all Tier 1 + Tier 2 metrics for one (dataset, object) pair.

    Parameters
    ----------
    vb_trajs     : VidBot trajectory samples — (N_vb, T, 3), camera-frame metres.
                   Note: all samples share a nearly identical start point because
                   VidBot anchors t=0 to the predicted contact location.
    da_trajs     : DA trajectory samples — (N_da, T, 3), camera-frame metres.
    vb_best_idx  : index into vb_trajs of the sample with lowest total guidance loss.

    Returns
    -------
    dict mapping metric name → float value
    """
    vb_best = vb_trajs[vb_best_idx]       # (T, 3) — VidBot's selected trajectory
    vb_mean = vb_trajs.mean(axis=0)       # (T, 3) — mean over all VidBot samples
    da_mean = da_trajs.mean(axis=0)       # (T, 3) — mean over all DA samples

    # ── Tier 1 ───────────────────────────────────────────────────────────────

    # start_l2: distance between VidBot's contact point and DA's mean predicted start.
    # VidBot's best start ≈ its contact prediction; DA's mean start is the centroid
    # of its unconditioned samples.
    start_l2 = float(np.linalg.norm(vb_best[0] - da_mean[0]))

    # end_l2: analogous for the goal / final position.
    end_l2 = float(np.linalg.norm(vb_best[-1] - da_mean[-1]))

    # mean_traj_mse: average squared distance between the two mean curves at each
    # timestep.  Measures how different the representative trajectories are.
    per_step_sq = ((vb_mean - da_mean) ** 2).sum(axis=-1)   # (T,)
    mean_mse    = float(per_step_sq.mean())

    al_vb  = arc_length(vb_best)
    al_da  = arc_length(da_mean)
    ratio  = al_da / al_vb if al_vb > 1e-6 else float("nan")

    # ── Tier 2 ───────────────────────────────────────────────────────────────

    # diversity: average spread of the trajectory bundle across all timesteps.
    # Using all-T avoids the misleading near-zero VidBot start diversity that
    # arises from contact anchoring.
    div_vb = trajectory_diversity(vb_trajs)
    div_da = trajectory_diversity(da_trajs)

    # chamfer_end: how well do the two models' *goal* distributions overlap?
    ch = chamfer_endpoints(vb_trajs, da_trajs)

    # dtw: warping distance between the two mean curves — robust to timing shifts.
    dtw = dtw_distance(vb_mean, da_mean)

    return {
        "start_l2_m":        round(start_l2, 5),
        "end_l2_m":          round(end_l2,   5),
        "mean_traj_mse":     round(mean_mse, 6),
        "arc_length_vidbot": round(al_vb,    5),
        "arc_length_da":     round(al_da,    5),
        "arc_length_ratio":  round(ratio,    4),
        "diversity_vidbot":  round(div_vb,   5),
        "diversity_da":      round(div_da,   5),
        "chamfer_end_m":     round(ch,       5),
        "dtw_distance":      round(dtw,      4),
    }


# ── Dataset evaluation loop ───────────────────────────────────────────────────

METRICS = [
    "start_l2_m", "end_l2_m", "mean_traj_mse",
    "arc_length_vidbot", "arc_length_da", "arc_length_ratio",
    "diversity_vidbot", "diversity_da", "chamfer_end_m", "dtw_distance",
]

METRIC_LABELS = {
    "start_l2_m":        "Start-point L2 (m)",
    "end_l2_m":          "End-point L2 (m)",
    "mean_traj_mse":     "Mean trajectory MSE (m²)",
    "arc_length_vidbot": "Arc length – VidBot (m)",
    "arc_length_da":     "Arc length – DA (m)",
    "arc_length_ratio":  "Arc length ratio (DA / VidBot)",
    "diversity_vidbot":  "Bundle diversity – VidBot (m)",
    "diversity_da":      "Bundle diversity – DA (m)",
    "chamfer_end_m":     "Chamfer dist. endpoints (m)",
    "dtw_distance":      "DTW distance (m)",
}


def run_eval(dataset_dir: Path) -> list[dict]:
    """
    Iterate over all IMG_* subfolders, match VidBot and DA prediction NPZs by
    object index, compute metrics, and return a list of row dicts.

    Skips objects where either model's NPZ is missing.
    Prints a one-line summary per object as it goes.
    """
    rows: list[dict] = []

    for ds_path in sorted(dataset_dir.glob("IMG_*")):
        sel_path = ds_path / "selection.json"
        if not sel_path.exists():
            continue

        with open(sel_path) as f:
            sel = json.load(f)
        instruction = sel.get("instruction", "")
        force_level = int(sel.get("force_level", -1))

        vb_pred_dir = ds_path / "prediction"
        da_pred_dir = ds_path / "diffuser_actor_prediction"
        if not vb_pred_dir.exists() or not da_pred_dir.exists():
            continue

        # Build object-index → DA file map
        da_by_obj: dict[int, Path] = {
            int(f.stem.split("_")[1]): f
            for f in sorted(da_pred_dir.glob("000000_*.npz"))
        }

        for vb_file in sorted(vb_pred_dir.glob("000000_*.npz")):
            obj_idx = int(vb_file.stem.split("_")[1])
            if obj_idx not in da_by_obj:
                continue

            vb = np.load(vb_file,             allow_pickle=True)
            da = np.load(da_by_obj[obj_idx],  allow_pickle=True)

            # Load trajectory arrays — remove batch dim if present
            vb_trajs = vb["pred_trajectories"]
            if vb_trajs.ndim == 4:
                vb_trajs = vb_trajs[0]   # (N_vb, T_vb, 3)

            da_trajs = da["pred_trajectories"]
            if da_trajs.ndim == 4:
                da_trajs = da_trajs[0]   # (N_da, 81, 3)

            # Resample VidBot to 81 timesteps if it differs (defensive, should be 81)
            if vb_trajs.shape[1] != 81:
                from scipy.interpolate import CubicSpline as _CS
                resampled = []
                for traj in vb_trajs:
                    t_in  = np.linspace(0.0, 1.0, len(traj))
                    t_out = np.linspace(0.0, 1.0, 81)
                    resampled.append(
                        np.stack([_CS(t_in, traj[:, d])(t_out)
                                  for d in range(3)], axis=1)
                    )
                vb_trajs = np.stack(resampled).astype(np.float32)

            # Identify VidBot's best sample via total guidance loss
            losses = vb.get("guide_losses-total_loss", None)
            if losses is not None:
                l = losses[0] if losses.ndim == 2 else losses
                best_idx = int(np.argmin(l))
            else:
                best_idx = 0

            try:
                m = compute_metrics(vb_trajs, da_trajs, best_idx)
            except Exception as exc:
                print(f"  WARN {ds_path.name} obj{obj_idx}: {exc}")
                continue

            row = {
                "dataset":     ds_path.name,
                "object_idx":  obj_idx,
                "instruction": instruction,
                "force_level": force_level,
                **m,
            }
            rows.append(row)
            print(f"  {ds_path.name} obj{obj_idx}: "
                  f"start_l2={m['start_l2_m']:.3f}m  "
                  f"end_l2={m['end_l2_m']:.3f}m  "
                  f"mse={m['mean_traj_mse']:.4f}  "
                  f"div_vb={m['diversity_vidbot']:.4f}  "
                  f"div_da={m['diversity_da']:.4f}  "
                  f"dtw={m['dtw_distance']:.2f}")

    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

# Colours and labels for force levels present in the dataset
FORCE_COLORS = {0: "#4C9BE8", 1: "#F28C38", 2: "#6EBF8B"}
FORCE_NAMES  = {0: "Level 0 – easy", 1: "Level 1 – medium", 2: "Level 2 – hard"}

TIER1_METRICS = [
    "start_l2_m", "end_l2_m", "mean_traj_mse",
    "arc_length_vidbot", "arc_length_da", "arc_length_ratio",
]
TIER2_METRICS = [
    "diversity_vidbot", "diversity_da", "chamfer_end_m", "dtw_distance",
]


def _boxplot_by_force(
    axes:         list,
    tier_metrics: list[str],
    rows:         list[dict],
    force_levels: list[int],
) -> None:
    """
    Draw one box-plus-scatter plot per metric, with one box per force level.
    Each dot represents a single (dataset, object) pair.
    """
    rng = np.random.default_rng(42)

    for ax, metric in zip(axes, tier_metrics):
        data = {
            fl: [r[metric] for r in rows
                 if r["force_level"] == fl and not np.isnan(r[metric])]
            for fl in force_levels
        }
        positions = list(range(len(force_levels)))

        bp = ax.boxplot(
            [data[fl] for fl in force_levels],
            positions   = positions,
            patch_artist = True,
            widths       = 0.5,
            medianprops  = dict(color="black", linewidth=2),
        )
        for patch, fl in zip(bp["boxes"], force_levels):
            patch.set_facecolor(FORCE_COLORS[fl])
            patch.set_alpha(0.75)

        # Overlay individual data points with horizontal jitter
        for pos, fl in zip(positions, force_levels):
            vals = data[fl]
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                [pos + j for j in jitter], vals,
                color=FORCE_COLORS[fl], edgecolors="black",
                linewidths=0.5, s=30, zorder=3, alpha=0.85,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [FORCE_NAMES[fl] for fl in force_levels], fontsize=9
        )
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=8)
        ax.grid(axis="y", alpha=0.3)


def make_plots(rows: list[dict], out_dir: Path) -> None:
    """
    Generate four figures and save them to out_dir.

    Figure 1 — tier1_by_force_level.png
        Box plots of Tier-1 metrics (start/end L2, MSE, arc length)
        split by force level 0 vs 1.

    Figure 2 — tier2_by_force_level.png
        Box plots of Tier-2 metrics (diversity, Chamfer, DTW)
        split by force level.

    Figure 3 — per_dataset_key_metrics.png
        Bar chart of four key metrics averaged per dataset,
        coloured by force level.

    Figure 4 — diversity_vs_agreement.png
        Scatter plots exploring whether higher diversity correlates with
        worse cross-model agreement.
    """
    out_dir.mkdir(exist_ok=True)

    # Only plot force levels 0 and 1 (level 2 has only 1 dataset — too few for stats)
    force_levels = sorted(
        fl for fl in {r["force_level"] for r in rows} if fl in (0, 1)
    )

    # ── Figure 1 & 2: box plots ───────────────────────────────────────────────
    for tier_name, tier_metrics in [
        ("tier1", TIER1_METRICS),
        ("tier2", TIER2_METRICS),
    ]:
        n = len(tier_metrics)
        ncols = (n + 1) // 2
        fig, axes = plt.subplots(
            2, ncols,
            figsize=(5 * ncols, 9),
            facecolor="white",
        )
        axes_flat = axes.flatten()

        fig.suptitle(
            f"VidBot vs 3D Diffuser Actor — {tier_name.upper()} metrics\n"
            f"stratified by force level  (0 = easy, 1 = medium)",
            fontsize=14, fontweight="bold",
        )
        _boxplot_by_force(axes_flat, tier_metrics, rows, force_levels)

        # Hide unused subplots
        for ax in axes_flat[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        path = out_dir / f"{tier_name}_by_force_level.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {path.name}")

    # ── Figure 3: per-dataset bar chart ───────────────────────────────────────
    key_metrics = ["start_l2_m", "end_l2_m", "mean_traj_mse", "dtw_distance"]
    datasets    = sorted({r["dataset"] for r in rows})
    ds_force    = {r["dataset"]: r["force_level"] for r in rows}

    fig, axes = plt.subplots(
        len(key_metrics), 1,
        figsize=(max(10, len(datasets) * 0.75), 4 * len(key_metrics)),
        facecolor="white",
    )
    fig.suptitle(
        "Per-dataset key metrics — VidBot vs 3D Diffuser Actor\n"
        "(bars = mean over objects; error bars = ±1 std)",
        fontsize=13, fontweight="bold",
    )

    for ax, metric in zip(axes, key_metrics):
        means, stds, colors = [], [], []
        for ds in datasets:
            vals = [r[metric] for r in rows
                    if r["dataset"] == ds and not np.isnan(r[metric])]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals))   if len(vals) > 1 else 0.0)
            colors.append(FORCE_COLORS.get(ds_force.get(ds, -1), "#aaaaaa"))

        x = np.arange(len(datasets))
        ax.bar(x, means, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    ecolor="black", elinewidth=1, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Force-level legend
        present_levels = sorted(
            fl for fl in FORCE_COLORS if fl in ds_force.values()
        )
        handles = [plt.Rectangle((0, 0), 1, 1,
                                  fc=FORCE_COLORS[fl], alpha=0.8)
                   for fl in present_levels]
        ax.legend(handles, [FORCE_NAMES[fl] for fl in present_levels],
                  fontsize=8, loc="upper right")

    plt.tight_layout()
    path = out_dir / "per_dataset_key_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path.name}")

    # ── Figure 4: diversity vs agreement scatter ──────────────────────────────
    #
    # Do more diverse (spread-out) trajectory bundles correspond to worse
    # cross-model agreement?  We test two hypotheses:
    #   Left:  VidBot bundle diversity → start-point L2 (contact agreement)
    #   Right: DA bundle diversity     → DTW distance  (overall curve agreement)

    scatter_pairs = [
        ("diversity_vidbot", "start_l2_m",   "VidBot bundle diversity → contact L2"),
        ("diversity_da",     "dtw_distance",  "DA bundle diversity → DTW distance"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
    fig.suptitle(
        "Does trajectory diversity predict cross-model disagreement?",
        fontsize=13, fontweight="bold",
    )

    for ax, (x_m, y_m, title) in zip(axes, scatter_pairs):
        for fl in force_levels:
            subset = [r for r in rows
                      if r["force_level"] == fl
                      and not np.isnan(r[x_m])
                      and not np.isnan(r[y_m])]
            if not subset:
                continue
            xs = np.array([r[x_m] for r in subset])
            ys = np.array([r[y_m] for r in subset])
            ax.scatter(xs, ys, c=FORCE_COLORS[fl], label=FORCE_NAMES[fl],
                       edgecolors="black", linewidths=0.5, s=60, alpha=0.85)

            # Pearson r annotation per force level
            if len(xs) > 2:
                r_val = float(np.corrcoef(xs, ys)[0, 1])
                ax.annotate(
                    f"r={r_val:.2f} (FL{fl})",
                    xy=(0.05, 0.95 - 0.07 * fl),
                    xycoords="axes fraction",
                    color=FORCE_COLORS[fl], fontsize=9,
                )

        ax.set_xlabel(METRIC_LABELS[x_m], fontsize=10)
        ax.set_ylabel(METRIC_LABELS[y_m], fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "diversity_vs_agreement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", default="./datasets",
        help="Root directory containing IMG_* dataset folders (default: ./datasets)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_csv     = dataset_dir / "eval_trajectories.csv"
    out_plots   = dataset_dir / "eval_plots"

    print("Computing metrics …\n")
    rows = run_eval(dataset_dir)

    if not rows:
        print("No matched VidBot + DA predictions found.")
        return

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = ["dataset", "object_idx", "instruction", "force_level"] + METRICS
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {out_csv}")

    # ── Print mean summary by force level ─────────────────────────────────────
    print("\n── Mean ± std by force level ───────────────────────────────────────")
    for fl in sorted({r["force_level"] for r in rows}):
        subset = [r for r in rows if r["force_level"] == fl]
        print(f"\n  Force level {fl}  ({FORCE_NAMES.get(fl,'?')})  "
              f"n={len(subset)} objects:")
        for m in METRICS:
            vals = [r[m] for r in subset if not np.isnan(r[m])]
            if vals:
                print(f"    {METRIC_LABELS[m]:<35} "
                      f"{np.mean(vals):.5f}  ±{np.std(vals):.5f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    make_plots(rows, out_plots)
    print(f"\nAll plots saved → {out_plots}/")


if __name__ == "__main__":
    main()

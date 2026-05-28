#!/usr/bin/env python3
"""Comprehensive training run analysis for RAPiDock fine-tuning.

Reads training_history.csv files from one or more phase output directories,
produces per-phase diagnostic tables, detects instability events, and optionally
generates matplotlib plots.

Usage:
    # Analyse current v1 run
    conda run -n score-env python scripts/analyze_training.py \
        --phase-dirs \
            third_party/RAPiDock_finetuned/finetune_peppc_phase1 \
            third_party/RAPiDock_finetuned/finetune_peppc_phase2 \
            third_party/RAPiDock_finetuned/finetune_peppc_phase3 \
        --out-dir logs/analysis_v1

    # Compare v1 vs v2
    conda run -n score-env python scripts/analyze_training.py \
        --phase-dirs \
            third_party/RAPiDock_finetuned/finetune_peppc_v2_phase1 \
            third_party/RAPiDock_finetuned/finetune_peppc_v2_phase2 \
            third_party/RAPiDock_finetuned/finetune_peppc_v2_phase3 \
        --compare-dirs \
            third_party/RAPiDock_finetuned/finetune_peppc_phase1 \
            third_party/RAPiDock_finetuned/finetune_peppc_phase2 \
            third_party/RAPiDock_finetuned/finetune_peppc_phase3 \
        --out-dir logs/analysis_comparison
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_history(phase_dir: str) -> Optional[List[dict]]:
    """Load training_history.csv from a phase output directory."""
    path = Path(phase_dir) / "training_history.csv"
    if not path.exists():
        return None
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


# ---------------------------------------------------------------------------
# Instability detection
# ---------------------------------------------------------------------------

BLOWUP_THRESHOLD = 1e6   # val_loss above this = full blowup
SPIKE_RATIO = 5.0         # val > spike_ratio × trimmed baseline = spike


def detect_instability(history: List[dict]) -> Dict[str, list]:
    """Identify pathological epochs."""
    # Skip epoch 1 (EMA cold start always inflated)
    stable = [r for r in history if r["epoch"] > 1 and math.isfinite(r.get("val_loss", float("nan")))]
    if not stable:
        return {}

    # Compute a "baseline" using the bottom 50% of val losses (trimmed reference)
    sorted_vals = sorted(r["val_loss"] for r in stable)
    half = max(1, len(sorted_vals) // 2)
    baseline = sum(sorted_vals[:half]) / half

    blowups   = [r for r in stable if r.get("val_loss", 0) > BLOWUP_THRESHOLD]
    spikes    = [r for r in stable if r.get("val_loss", 0) > baseline * SPIKE_RATIO
                 and r.get("val_loss", 0) <= BLOWUP_THRESHOLD]
    norm_jumps = []
    for i in range(1, len(history)):
        prev = history[i - 1].get("tr_norm_train", 0.0)
        cur  = history[i].get("tr_norm_train", 0.0)
        if prev > 0 and cur > prev * 3.0:
            norm_jumps.append({
                "epoch": history[i]["epoch"],
                "prev_norm": prev,
                "cur_norm": cur,
                "ratio": cur / prev,
            })
    return {"blowups": blowups, "spikes": spikes, "norm_jumps": norm_jumps,
            "baseline": baseline}


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize(history: List[dict], label: str) -> dict:
    """Compute per-phase summary statistics."""
    if not history:
        return {}
    # Skip EMA cold start epoch 1
    valid = [r for r in history if r["epoch"] > 1]
    if not valid:
        valid = history

    val_losses   = [r["val_loss"] for r in valid if math.isfinite(r.get("val_loss", float("nan")))]
    train_losses = [r["train_loss"] for r in history]

    best_row = min(valid, key=lambda r: r.get("val_loss", float("inf")))

    # Fraction of epochs with val > 1000 (instability rate)
    n_bad = sum(1 for v in val_losses if v > 1000.0)

    return {
        "label":           label,
        "n_epochs":        len(history),
        "best_val":        best_row.get("val_loss", float("nan")),
        "best_epoch":      int(best_row.get("epoch", 0)),
        "final_train":     train_losses[-1] if train_losses else float("nan"),
        "min_train":       min(train_losses) if train_losses else float("nan"),
        "val_instab_pct":  100.0 * n_bad / max(len(val_losses), 1),
        "mean_tr_norm":    sum(r.get("tr_norm_train", 0) for r in history) / max(len(history), 1),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_summary_table(summaries: List[dict]) -> None:
    header = (f"{'Phase':<30} {'Best Val':>10} {'@Ep':>5} {'Final Train':>12} "
              f"{'Instab%':>8} {'Avg tr|norm|':>13}")
    print("\n" + "=" * 85)
    print("TRAINING SUMMARY")
    print("=" * 85)
    print(header)
    print("-" * 85)
    for s in summaries:
        if not s:
            continue
        print(f"{s['label']:<30} {s['best_val']:>10.4f} {s['best_epoch']:>5d} "
              f"{s['final_train']:>12.4f} {s['val_instab_pct']:>7.1f}% "
              f"{s['mean_tr_norm']:>13.4f}")
    print("=" * 85)


def print_instability_report(label: str, instab: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"INSTABILITY REPORT: {label}")
    print(f"{'─'*60}")
    n_blowup = len(instab.get("blowups", []))
    n_spike  = len(instab.get("spikes", []))
    n_norm   = len(instab.get("norm_jumps", []))
    baseline = instab.get("baseline", float("nan"))
    print(f"  val baseline (bottom-50% median): {baseline:.4f}")
    print(f"  Full blowups (val > 1e6):          {n_blowup} epochs")
    print(f"  Spikes (val > {SPIKE_RATIO}× baseline): {n_spike} epochs")
    print(f"  tr_norm jumps (>3× prev epoch):    {n_norm} epochs")
    if instab.get("blowups"):
        for r in instab["blowups"][:5]:
            print(f"    BLOWUP epoch {int(r['epoch'])}: val={r['val_loss']:.3e}  "
                  f"tr_norm_max={r.get('val_tr_norm_max', 0):.1f}")
    if instab.get("norm_jumps"):
        for j in instab["norm_jumps"][:5]:
            print(f"    NORM JUMP epoch {int(j['epoch'])}: "
                  f"{j['prev_norm']:.3f} → {j['cur_norm']:.3f} (×{j['ratio']:.1f})")


def print_epoch_table(history: List[dict], label: str, n_cols=20) -> None:
    """Print a condensed epoch-by-epoch table."""
    print(f"\n{label} — epoch table (every epoch):")
    print(f"  {'Ep':>4}  {'TrainLoss':>10}  {'ValLoss(trim)':>14}  "
          f"{'ValMedian':>10}  {'tr|norm|':>9}  {'LR':>9}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*9}  {'─'*9}")
    for r in history:
        ep       = int(r.get("epoch", 0))
        tr       = r.get("train_loss", float("nan"))
        val      = r.get("val_loss",   float("nan"))
        med      = r.get("val_median", float("nan"))
        nrm      = r.get("tr_norm_train", 0.0)
        lr       = r.get("lr", float("nan"))
        flag     = " ⚠" if (math.isfinite(val) and val > 1000.0) else ""
        print(f"  {ep:>4}  {tr:>10.4f}  {val:>14.4f}  "
              f"{med:>10.4f}  {nrm:>9.4f}  {lr:>9.2e}{flag}")


# ---------------------------------------------------------------------------
# Optional matplotlib plots
# ---------------------------------------------------------------------------

def try_plot(histories: List[tuple], out_dir: Path) -> None:
    """Generate loss + norm plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: val loss curves (trimmed mean) ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, history in histories:
        epochs = [r["epoch"] for r in history]
        val_trim = [r.get("val_loss", float("nan")) for r in history]
        val_med  = [r.get("val_median", float("nan")) for r in history]
        train    = [r.get("train_loss", float("nan")) for r in history]

        # Cap at 1000 for visibility (blowups would squash everything else)
        val_trim_clip = [min(v, 1000.0) if math.isfinite(v) else 1000.0 for v in val_trim]
        val_med_clip  = [min(v, 1000.0) if math.isfinite(v) else 1000.0 for v in val_med]

        axes[0].plot(epochs, val_trim_clip, label=f"{label} val(trim, capped@1k)")
        axes[0].plot(epochs, val_med_clip,  label=f"{label} val(median)", linestyle="--", alpha=0.6)
        axes[1].plot(epochs, train,         label=f"{label} train")

    axes[0].set_title("Validation loss (trimmed mean + median, capped at 1000)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    loss_path = out_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=120)
    plt.close()
    print(f"[analyze] Plot saved: {loss_path}")

    # ── Plot 2: score norms ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, history in histories:
        epochs  = [r["epoch"] for r in history]
        tr_norm = [r.get("tr_norm_train", 0.0) for r in history]
        ax.plot(epochs, tr_norm, label=f"{label} tr|norm|(train)")

    ax.set_title("Translation score norm (train model) — instability detector")
    ax.set_xlabel("Epoch"); ax.set_ylabel("||tr_pred||_2 mean")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    norm_path = out_dir / "score_norms.png"
    plt.tight_layout()
    plt.savefig(norm_path, dpi=120)
    plt.close()
    print(f"[analyze] Plot saved: {norm_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyze RAPiDock fine-tuning runs.")
    p.add_argument("--phase-dirs", nargs="+", required=True,
                   help="Phase output directories to analyze (primary run)")
    p.add_argument("--compare-dirs", nargs="+", default=[],
                   help="Optional: second run dirs for head-to-head comparison")
    p.add_argument("--out-dir", default="logs/analysis",
                   help="Directory to write reports and plots to")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib plot generation")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load histories ────────────────────────────────────────────────────
    def load_all(dirs, prefix):
        results = []
        for i, d in enumerate(dirs, start=1):
            h = load_history(d)
            label = f"{prefix} P{i} ({Path(d).name[:30]})"
            if h is None:
                print(f"[analyze] WARNING: no training_history.csv in {d}")
            results.append((label, h))
        return results

    primary  = load_all(args.phase_dirs,    "primary")
    compare  = load_all(args.compare_dirs,  "compare") if args.compare_dirs else []

    all_runs = [(lbl, h) for lbl, h in primary + compare if h is not None]

    if not all_runs:
        print("[analyze] No training histories found. Check --phase-dirs.")
        sys.exit(1)

    # ── Per-phase reports ─────────────────────────────────────────────────
    summaries = []
    for label, history in all_runs:
        print_epoch_table(history, label)
        instab = detect_instability(history)
        print_instability_report(label, instab)
        summaries.append(summarize(history, label))

    print_summary_table(summaries)

    # ── Comparison verdict ────────────────────────────────────────────────
    if len(all_runs) >= 2:
        primary_sum  = [s for s in summaries if s["label"].startswith("primary")]
        compare_sum  = [s for s in summaries if s["label"].startswith("compare")]
        if primary_sum and compare_sum:
            best_primary = min(s["best_val"] for s in primary_sum if s)
            best_compare = min(s["best_val"] for s in compare_sum if s)
            print(f"\n{'='*60}")
            print("COMPARISON VERDICT")
            print(f"  Primary best val:  {best_primary:.4f}")
            print(f"  Compare best val:  {best_compare:.4f}")
            delta = best_compare - best_primary
            if delta < 0:
                print(f"  → Primary is BETTER by {-delta:.4f} ({-100*delta/max(best_compare,1e-9):.1f}%)")
            else:
                print(f"  → Compare is BETTER by {delta:.4f} ({100*delta/max(best_primary,1e-9):.1f}%)")
            print(f"{'='*60}")

    # ── Save CSV summary ──────────────────────────────────────────────────
    summary_path = out_dir / "summary.csv"
    if summaries:
        fields = list(summaries[0].keys())
        with open(summary_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(s for s in summaries if s)
        print(f"\n[analyze] Summary CSV: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        try_plot(all_runs, out_dir)


if __name__ == "__main__":
    main()

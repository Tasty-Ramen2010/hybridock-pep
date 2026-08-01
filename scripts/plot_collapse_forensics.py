#!/usr/bin/env python3
"""plot_collapse_forensics.py — visualize the finetune_normloss BatchNorm collapse.

Renders the forensic evidence that the fine-tune collapse was driven by BatchNorm
running-statistic drift (not the loss weighting, not the score head). Data extracted
from the finetune_normloss checkpoints (base / ep1 / ep19 / ep22 / ep28) and the
N=8 per-epoch docking meter (logs/normloss_epoch_eval.log). Numbers are hard-coded
so the figure regenerates without the (multi-GB) checkpoints present.

Output: docs/collapse_forensics.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- BN running_var norm trajectory (cross-attention / placement convs) ----------
EPX = [0, 1, 19, 22, 28]                       # x-axis (0 = pretrained base)
CROSS_BN_VAR = {
    "cross_convs.0": [812822, 432944, 507507, 486942, 383443],
    "cross_convs.1": [121699,  57962,  68339,  67127, 648115],
    "cross_convs.2": [181942,  91865, 122056, 129835, 1412316],
    "cross_convs.3": [ 99119,  44541,  48180,  53014, 3015490],   # deepest = worst (30x)
}
# --- docking RMSD (best-of-N=8 DIRECT) per length/SS class, vs epoch --------------
BASE = {"short5": 2.06, "med9": 4.36, "long15": 5.12, "vlong28": 9.42, "SHEET12": 5.23}
MET_EP = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
MET = {
    "short5":  [2.86, 2.96, 3.00, 3.85, 3.51, 3.57, 5.11, 10.69, 18.52, 7.62],
    "med9":    [4.19, 3.09, 5.18, 3.59, 4.57, 8.73, 9.52,  7.92, 12.18, 7.57],
    "long15":  [4.42, 5.30, 6.10, 5.07, 9.07, 15.68, 11.64, 6.62, 13.97, 11.80],
    "vlong28": [11.52, 8.94, 9.74, 8.49, 11.53, 17.39, 8.75, 10.88, 17.65, 15.09],
    "SHEET12": [6.46, 6.86, 6.50, 5.39, 6.48, 7.39, 13.08, 8.89, 17.81, 11.03],
}
# --- drift decomposition & head shrinkage ----------------------------------------
DRIFT = {"learnable\nweights+biases": 0.0233, "BN running\nbuffers": 2.1566}
TRFINAL_W = [0.1639, 0.1635, 0.1098, 0.1018, 0.0876]   # tr_final output-linear weight norm

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle("finetune_normloss collapse forensics — BatchNorm running-stat drift is the driver",
             fontsize=13, fontweight="bold")

# (A) BN running_var trajectory (log scale) with collapse band
a = ax[0, 0]
for k, v in CROSS_BN_VAR.items():
    a.plot(EPX, v, marker="o", label=k)
a.axvspan(22, 28, color="red", alpha=0.08)
a.text(25, a.get_ylim()[1], "docking\ncollapse", ha="center", va="top", color="red", fontsize=8)
a.set_yscale("log"); a.set_xlabel("epoch (0 = pretrained)"); a.set_ylabel("‖BN running_var‖ (log)")
a.set_title("(A) Cross-attention BN running_var explodes 5–30× at ep22→28"); a.legend(fontsize=7)

# (B) docking RMSD collapse per class
b = ax[0, 1]
for k, v in MET.items():
    b.plot(MET_EP, v, marker="s", label=k)
    b.axhline(BASE[k], ls=":", lw=0.8, alpha=0.4, color=b.lines[-1].get_color())
b.axvspan(22, 28, color="red", alpha=0.08)
b.set_xlabel("epoch"); b.set_ylabel("best-of-8 DIRECT RMSD (Å)")
b.set_title("(B) Docking collapses exactly as BN var explodes\n(dotted = pretrained baseline)")
b.legend(fontsize=7)

# (C) where the model actually changed
c = ax[1, 0]
bars = c.bar(list(DRIFT.keys()), list(DRIFT.values()),
             color=["#4c72b0", "#c44e52"])
c.set_ylabel("relative drift ‖ΔW‖ / ‖W₀‖ (ep28 vs base)")
c.set_title("(C) ~99% of the change is BN buffers, not weights")
for bar, val in zip(bars, DRIFT.values()):
    c.text(bar.get_x() + bar.get_width()/2, val, f"{val:.3f}", ha="center", va="bottom")

# (D) translation magnitude head shrinks (secondary effect)
d = ax[1, 1]
d.plot(EPX, TRFINAL_W, marker="D", color="#55a868")
d.set_xlabel("epoch (0 = pretrained)"); d.set_ylabel("‖tr_final_layer output weight‖")
d.set_title("(D) Translation magnitude head shrinks −46%\n(head adapts to a moving BN target)")
d.set_ylim(0, 0.18)

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "docs/collapse_forensics.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")

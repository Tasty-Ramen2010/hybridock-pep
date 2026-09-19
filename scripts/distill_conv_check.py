import torch
BUF = ("running_mean", "running_var", "num_batches_tracked", "output_mask")
def sd(f): return torch.load(f, map_location="cpu", weights_only=False)["model"]
P = sd("third_party/RAPiDock_finetuned/longft_tanh/rapidock_finetuned_epoch010.pt")

def group(C, pref):
    """Relative change over TRAINABLE weights only.

    BatchNorm running_var under cross_convs has norm ~812,822 against fc.0.weight's 115, and
    --freeze-bn-stats pins it so its delta is exactly 0. Including buffers in the denominator
    therefore divides a real weight change by a frozen number 4 orders of magnitude larger and
    reports ~1e-05 no matter what the weights did. That is what produced the earlier
    "the convolutions never trained" reading.
    """
    dn = wn = 0.0
    for k in P:
        if not k.startswith(pref) or k not in C: continue
        if any(b in k for b in BUF): continue
        if P[k].shape != C[k].shape or not P[k].dtype.is_floating_point: continue
        if float(P[k].float().norm()) == 0.0: continue
        dn += float((C[k].float() - P[k].float()).norm()) ** 2
        wn += float(P[k].float().norm()) ** 2
    return (dn ** 0.5) / max(wn ** 0.5, 1e-12)

prefs = ["encoder.cross_convs", "encoder.intra_convs", "encoder.cross_type_embedding",
         "encoder.tr_final_layer"]
print(f"{'arm':<24}{'cross_convs':>13}{'intra_convs':>13}{'embedding':>13}{'gate':>10}")
for tag, f in (("run1 ep01 (lr 1e-5)", "distill_v1/rapidock_finetuned_epoch001.pt"),
               ("run1 ep15 (lr 1e-5)", "distill_v1/rapidock_finetuned_epoch015.pt"),
               ("run2 ep01 (lr 1e-4)", "distill_v2/rapidock_finetuned_epoch001.pt"),
               ("run2 ep08 (lr 1e-4)", "distill_v2/rapidock_finetuned_epoch008.pt"),
               ("run2 ep15 (lr 1e-4)", "distill_v2/rapidock_finetuned_epoch015.pt")):
    try:
        C = sd("third_party/RAPiDock_finetuned/" + f)
    except FileNotFoundError:
        continue
    v = [group(C, p) for p in prefs]
    print(f"{tag:<24}{v[0]:>13.3e}{v[1]:>13.3e}{v[2]:>13.3e}{v[3]:>10.1e}")

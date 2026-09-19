#!/usr/bin/env bash
# Build the DOCKING learning curve for the from-scratch balanced run.
#
# Why this exists: val loss is anti-correlated with docking quality on this
# architecture (measured Sep-09), and RAPiDock's own config selects on
# `valinf_rmsds_backbone_lt2` -- real docking success -- not loss. Their shipped
# checkpoint is epoch 199 of 850 for exactly that reason. So the only trustworthy
# progress signal is periodic direct-RMSD benchmarking of saved checkpoints.
#
# Runs on LOCAL (idle) so it never competes with the two training GPUs. Pulls the
# newest checkpoint from rtx6000 every INTERVAL epochs, benchmarks it on the same
# 15 complexes / N=24 protocol as the Sep-09 five-way bench, and appends one row to
# a CSV. Reference points on that identical protocol:
#     pretrained  mean 4.68A  <=2A 3/15  <=5A 12/15  spread 8.29A
#     ep63        mean 21.05A <=2A 0/15  <=5A  0/15  spread 63.77A
# SPREAD is the early diagnostic: it must fall from ~64A toward ~8A as the model
# learns to localise at all. If it is still huge by ~ep200, placement is not being
# learned and the run should be cut rather than left to burn 10 more days.
set -u
REPO=/home/igem/unknown_software
REMOTE=rtx6000
RDIR=~/Ram_Work/hybridock-pep/third_party/RAPiDock_finetuned/scratch850_balanced_s43
INTERVAL=${1:-40}
OUT=$REPO/logs/scratch_curve.csv
EVAL=$REPO/third_party/RAPiDock_finetuned/eval_curve
export RAPIDOCK_PY=/home/igem/miniconda3/envs/rapidock/bin/python
export OMP_NUM_THREADS=1

mkdir -p "$EVAL" "$REPO/runs"
[ -f "$OUT" ] || echo "epoch,mean_rmsd,median,le2A,le5A,spread,timestamp" > "$OUT"
cp $REPO/third_party/RAPiDock_finetuned/train_models/CGTensorProductEquivariantModel/model_parameters.yml "$EVAL/" 2>/dev/null

last_done=0
while true; do
  EP=$(ssh -o ControlPath=none $REMOTE "ls $RDIR/*.pt 2>/dev/null | grep -oE 'epoch[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1" 2>/dev/null)
  if [ -n "${EP:-}" ] && [ "$EP" -ge $((last_done + INTERVAL)) ]; then
    CK="rapidock_finetuned_epoch${EP}.pt"
    SZ=$(ssh -o ControlPath=none $REMOTE "stat -c%s $RDIR/$CK" 2>/dev/null)
    rm -f "$EVAL"/*.pt
    # rsync, no timeout wrapper: a truncated checkpoint fails as a corrupt zip and
    # wastes a whole benchmark run (happened once already).
    rsync -a --partial --inplace -e "ssh -o ControlPath=none" "$REMOTE:$RDIR/$CK" "$EVAL/" 2>/dev/null
    GOT=$(stat -c%s "$EVAL/$CK" 2>/dev/null || echo 0)
    if [ "$GOT" != "${SZ:-x}" ]; then
      echo "[curve] ep$EP transfer incomplete ($GOT/$SZ) — will retry next cycle"
      sleep 600; continue
    fi
    echo "[curve] benchmarking epoch $EP ..."
    RES=$($RAPIDOCK_PY $REPO/scripts/exp_runner.py \
        --bench $REPO/data/bench15_local.csv --limit 15 --n 24 \
        --infer-dir $REPO/third_party/RAPiDock_finetuned \
        --model-dir "$EVAL" --ckpt "$CK" \
        --out $REPO/runs/curve_ep$EP --partial-mode fixed --partial 1:1:1 \
        --label curve_ep$EP 2>/dev/null | grep -E "^n=" | tail -1)
    if [ -n "$RES" ]; then
      M=$(echo "$RES"  | grep -oE 'mean=[0-9.]+'   | cut -d= -f2)
      MD=$(echo "$RES" | grep -oE 'median=[0-9.]+' | cut -d= -f2)
      L2=$(echo "$RES" | grep -oE 'le2A=[0-9]+/[0-9]+' | cut -d= -f2)
      L5=$(echo "$RES" | grep -oE 'le5A=[0-9]+/[0-9]+' | cut -d= -f2)
      SP=$(echo "$RES" | grep -oE 'mean_spread=[0-9.]+' | cut -d= -f2)
      echo "$EP,$M,$MD,$L2,$L5,$SP,$(date '+%m-%d %H:%M')" >> "$OUT"
      echo "[curve] ep$EP -> mean=${M}A spread=${SP}A le2A=$L2"
    fi
    rm -rf $REPO/runs/curve_ep$EP        # poses are large; the CSV row is what we keep
    last_done=$EP
  fi
  sleep 900
done

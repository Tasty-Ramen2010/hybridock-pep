# Overnight session log — 2026-06-13 (Ram asleep)

**Mandate:** extract PDBbind/datasets from Ram's Google Drive (choppapurandhar@gmail.com), build a
physics+ML combo + train our own models to BEAT PPI-Affinity (0.554 on T100), diagnose why our RMSE is
"so bad," log all activity. Tokens free; commit incrementally; honest evaluation discipline (physics
never lies; no sign-fitting; cross-dataset sign-stability gate; LOO not in-sample; small-n = suspect).

Ram's framing to validate: PPI-Affinity is evaluated on CRYSTAL structures (no AI poses). On crystal we
already tie it (0.544 vs 0.554) → we beat/match it AT ITS OWN EVALUATION; it would suffer the same
AI-pose haircut we do if it scored RAPiDock poses. The AI-pose robustness is OUR added scope, not a deficit.

## Activity timeline
- **04:42** session start. Loaded Google Drive MCP tools. Starting dataset hunt.
- **04:45** ⚠️ BLOCKER: Google Drive MCP connector failed to stay connected (tried list/search; server
  disconnected). Cannot reach Ram's PDBbind uploads tonight — needs the Drive integration reconnected
  (a UI/auth step) or a shared link with file IDs. Logged for Ram; NOT burning the night on it.
- **04:46** PIVOT: confirmed our pooled 156 contains **91 of PPI-Affinity's 100 peptide test complexes
  with IDENTICAL labels (mean |Δ|=0.0 kcal)**. → I can run the DEFINITIVE head-to-head (their published
  predictions vs ours, same complexes) and the RMSE diagnosis with zero Drive dependency. Building e103.
- **04:52** E103 RESULT (honest, sobering). On the SHARED 91 crystal complexes:
  - ours r=0.451 fitRMSE=1.55 | **PPI-Affinity r=0.629 fitRMSE=1.35** | Kdeep 0.32, DFIRE 0.35, CP_PIE
    −0.35, RF-Score 0.28, PRODIGY 0.12.
  - So on the apples-to-apples diverse subset, **PPI-Affinity genuinely beats us (Δr −0.18)** — the pooled
    "0.544≈0.554 parity" was buoyed by easier cr65 complexes not in this subset. We still crush every
    OTHER physics method. Ram's "we beat it at its own eval" → only true on the mixed pool, NOT the
    matched diverse subset. Must tell him straight.
  - **RMSE answer (Q):** RMSE ≈ std_y·√(1−r²), verified exactly (our full-156 std 2.14, r 0.544 → 1.80 ✓).
    Our "bad" RMSE = (a) wider affinity spread in our 156 (2.14 vs T100 1.73) + (b) genuinely lower r on
    the diverse subset. Not a separate calibration bug; it tracks r×spread.
  - THE BAR to beat: **0.629 on the shared 91.** That's the target for the physics+ML combo. Building e104.
- **04:58** E104 physics+ML combo — DECISIVE, honest. On n=156 LOCO:
  - M0 ridge/struct+router: pooled **0.587**, shared-91 **0.449** (BEST)
  - M1 GBT/struct 0.503/0.307 · M2 GBT/struct+seq 0.483/0.245 · M3 stack 0.520/0.304 — **all WORSE**.
  - Nonlinear ML + sequence features OVERFIT on 156 and HURT. Linear physics is permutation-genuine
    (null max 0.12 ≪ 0.449). → The gap to PPI-Affinity 0.629 is **DATA SCALE** (their 949 labeled
    complexes), NOT model cleverness. Confirms the PDBbind/Drive data lever is THE bottleneck.
  - Bonus: M0 with net_dewet+polar_desolv+router = 0.587 pooled, our best honest pooled number.
- **05:00** Next: test whether our physics is COMPLEMENTARY to PPI-Affinity (orthogonal → ensemble wins).
  This is a Drive-free, potentially strong positive ("use both / we add value even to the leader").

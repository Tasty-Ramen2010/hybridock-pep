# Physics-failure atlas — all protein-peptide complexes (n=951)

Composition: 156 ours + 795 PDBbind. By length: short≤8=292, med9-12=417, long13-16=168, vlong≥17=74.

## PART 1 — feature × length correlation  corr(feature, ΔG)  [sign-stable across bands = real physics]
feature           short≤8    med9-12  long13-16   vlong≥17      ALL  verdict
poc_n               -0.18      +0.00      +0.03      -0.24    -0.07  weak
poc_f_hyd           +0.10      -0.19      +0.01      -0.04    -0.08  FLIPS
poc_f_arom          -0.23      +0.04      -0.10      +0.06    -0.04  FLIPS
poc_net             -0.01      +0.03      -0.06      +0.19    +0.03  FLIPS
poc_eis             +0.02      -0.14      -0.09      -0.13    -0.08  STABLE
bsa_hyd             -0.07      -0.06      +0.16      -0.19    -0.06  FLIPS
sasa_hb             -0.24      -0.06      -0.11      -0.10    -0.11  STABLE
sasa_sb             -0.10      -0.09      -0.01      -0.20    -0.10  STABLE
arom_cc             -0.05      -0.07      +0.09      -0.06    -0.06  FLIPS
hb_count            -0.18      -0.14      -0.13      -0.15    -0.17  STABLE
strength_bur        -0.07      -0.23      +0.08      -0.19    -0.13  FLIPS
mean_burial         -0.17      +0.04      +0.06      -0.09    +0.04  FLIPS
mj_contact          +0.11      +0.05      -0.06      +0.26    +0.12  FLIPS
rg_per_L            +0.01      +0.30      +0.04      +0.23    +0.24  weak
org_density         -0.04      -0.38      -0.17      -0.22    -0.29  STABLE
cys_frac            -0.04      -0.09      -0.32      -0.16    -0.14  STABLE
abs_charge          -0.09      -0.07      -0.05      +0.01    -0.05  (derived)
|net_charge|        -0.02      -0.00      -0.03      +0.05    -0.05  (derived)
hyd_frac            -0.06      -0.10      -0.06      -0.15    -0.08  (derived)
length              -0.01      -0.12      +0.02      -0.28    -0.17  (derived)

## PART 2 — WHERE WE FAIL (GBT 5-fold, pooled r=+0.450 RMSE=1.75)
stratum                 n        r    RMSE  mean|err|
  len short≤8          292    +0.32    1.62       1.27
  len med9-12          417    +0.47    1.81       1.44
  len long13-16        168    +0.48    1.80       1.48
  len vlong≥17          74    +0.26    1.75       1.45
  charge ≤0.15       268    +0.44    1.83       1.49
  charge 0.15-0.30   357    +0.51    1.70       1.36
  charge >0.30       326    +0.40    1.73       1.35
  src ours           156    +0.52    1.84       1.50
  src pdbbind        795    +0.37    1.73       1.38

## PART 3 — WHICH PHYSICS IS MISSING  corr(|residual|, missing-term proxy) by length band
  (positive = error GROWS with that effect ⇒ that physics is absent from the model)
proxy                         short≤8    med9-12  long13-16   vlong≥17      ALL
conf-entropy(rg_per_L)          +0.00      -0.08      +0.03      +0.18    -0.06
conf-entropy(length)            +0.07      +0.01      -0.16      +0.10    +0.07
disorder(1-org_density)         -0.12      -0.06      -0.06      +0.21    -0.06
electrostatics(|netQ|)          -0.14      -0.03      -0.04      +0.16    -0.01
electrostatics(absQ)            -0.14      -0.06      +0.07      +0.14    -0.07
salt-bridge(sasa_sb)            -0.10      -0.07      +0.08      -0.10    -0.05
hydrophobic(hyd_fr)             +0.13      +0.09      -0.05      -0.10    +0.08

## PART 4 — WHY EVERYONE FAILS (shared-91: PPI/Kdeep/DFIRE/CP_PIE/RF/PRODIGY + ours)
  Per-method |z-error| correlation (do methods fail on the SAME complexes? high = shared hard cases):
   PPI-Affinity   +1.00  +0.07  +0.15  +0.11  +0.01  +0.05
   Kdeep          +0.07  +1.00  +0.30  +0.29  +0.15  +0.14
   DFIRE          +0.15  +0.30  +1.00  +0.94  +0.70  +0.46
   CP_PIE         +0.11  +0.29  +0.94  +1.00  +0.66  +0.39
   RF-Score       +0.01  +0.15  +0.70  +0.66  +1.00  +0.47
   PRODIGY        +0.05  +0.14  +0.46  +0.39  +0.47  +1.00
                 PPI-Af  Kdeep  DFIRE CP_PIE RF-Sco PRODIG

  TOP-10 hardest complexes (highest consensus error across all 6 methods = intrinsically hard):
   3r85A.pdb    y=-6.0  consensus|z-err|=2.61  (PPI 3.6, best other 1.3)
   1nyuB.pdb    y=-10.9  consensus|z-err|=2.34  (PPI 2.4, best other 1.0)
   2bypC.pdb    y=-12.4  consensus|z-err|=2.29  (PPI 0.3, best other 2.4)
   1nyuC.pdb    y=-10.9  consensus|z-err|=2.16  (PPI 1.2, best other 1.6)
   1ywhG.pdb    y=-10.6  consensus|z-err|=1.91  (PPI 0.5, best other 1.7)
   2hrpH.pdb    y=-11.3  consensus|z-err|=1.89  (PPI 1.7, best other 1.5)
   1osvB.pdb    y=-4.8  consensus|z-err|=1.86  (PPI 0.8, best other 1.7)
   3gsnH.pdb    y=-6.2  consensus|z-err|=1.78  (PPI 0.9, best other 0.8)
   3lk4D.pdb    y=-12.0  consensus|z-err|=1.77  (PPI 1.8, best other 0.9)
   2g9hD.pdb    y=-9.6  consensus|z-err|=1.65  (PPI 0.2, best other 1.2)

  mean consensus error: 0.90; if the hardest cases are high-charge/long/flexible,
  that regime needs DYNAMICS/FEP (no static method, incl PPI, captures it). Where PPI alone succeeds
  but physics methods fail = data-learnable (sequence statistics), not single-pose-computable.

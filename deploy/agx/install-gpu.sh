#!/bin/bash
# Runs INSIDE the AGX chroot. Builds the GPU sampling environment ("rapidock-gpu"): Python 3.8 + NVIDIA's Jetson PyTorch wheel
# (CUDA 11.4) + the CUDA libraries the chroot lacks + torch_scatter / torch_cluster compiled for the Orin (sm_87). Resumable.
set -uo pipefail
cd /mnt/hdp
mkdir -p dl cuda-extra
CONDA=/mnt/hdp/miniforge3
export PATH=$CONDA/bin:$PATH
REPO=https://repo.download.nvidia.com/jetson/common
say() { echo "[$(date -u +%T)] $*"; }
step() { [ -e "dl/.step-$1" ] && { say "skip step $1 ($2)"; return 1; }; say "STEP $1: $2"; return 0; }
done_() { touch "dl/.step-$1"; say "STEP $1 done"; }
fail() { say "FAILED: $*"; exit 1; }
dl() { local i; for i in 1 2 3 4 5 6 7 8; do curl -fsS -C - -m 1500 -o "$1" "$2" && return 0; say "  retry $i for $(basename "$1")"; sleep 4; done; return 1; }

if step 0 "free space (conda cache)"; then conda clean --all -y >/dev/null 2>&1; df -h /mnt/hdp | tail -1; done_ 0; fi

if step 1 "CUDA 11.4 libraries and headers the chroot lacks (Jetson L4T packages, unpacked, nothing installed system-wide)"; then
  curl -fsS -m 300 $REPO/dists/r35.4/main/binary-arm64/Packages.gz | gunzip > dl/Packages || fail "package index"
  for pkg in libcudnn8 libcusparse-11-4 libcusparse-dev-11-4 libcusolver-11-4 libcusolver-dev-11-4 libcurand-11-4 libcurand-dev-11-4 \
             libcufft-11-4 libcufft-dev-11-4 cuda-nvtx-11-4 cuda-nvrtc-11-4 cuda-nvrtc-dev-11-4 cuda-cupti-11-4; do
    [ -e "dl/.pkg-$pkg" ] && { say "  already have $pkg"; continue; }
    fn=$(awk -v p="$pkg" '$1=="Package:"{cur=$2} $1=="Filename:"&&cur==p{print $2; exit}' dl/Packages)
    sha=$(awk -v p="$pkg" '$1=="Package:"{cur=$2} $1=="SHA256:"&&cur==p{print $2; exit}' dl/Packages)
    [ -n "$fn" ] || { say "not in index: $pkg"; continue; }
    f=dl/$(basename "$fn")
    dl "$f" "$REPO/$fn" || fail "download $pkg"
    echo "$sha  $f" | sha256sum -c --quiet - || fail "checksum $pkg"
    dpkg -x "$f" cuda-extra && rm -f "$f" && touch "dl/.pkg-$pkg"
    say "  unpacked $pkg"
  done
  du -sh cuda-extra; df -h /mnt/hdp | tail -1
  done_ 1
fi
X=/mnt/hdp/cuda-extra
CUDALIB=$(ls -d $X/usr/local/cuda-11.4/targets/*/lib | head -1)
CUDAINC=$(ls -d $X/usr/local/cuda-11.4/targets/*/include | head -1)
CUDNNLIB=$X/usr/lib/aarch64-linux-gnu
export LD_LIBRARY_PATH=$CUDALIB:$CUDNNLIB:/usr/local/tegra-libs:/usr/local/cuda-11.4/lib64
say "extra libs: $CUDALIB | headers: $CUDAINC"

if step 2 "python 3.8 environment 'rapidock-gpu'"; then
  conda create -y -n rapidock-gpu -c conda-forge python=3.8 pip 'numpy<2' scipy scikit-learn networkx pandas tqdm psutil pyyaml sympy matplotlib > dl/conda-gpu.log 2>&1 || { tail -20 dl/conda-gpu.log; fail "conda create"; }
  done_ 2
fi
PY=$CONDA/envs/rapidock-gpu/bin/python

if step 3 "Jetson PyTorch wheel (torch 2.0.0, CUDA 11.4)"; then
  W=torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
  dl "dl/$W" "https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/$W" || fail "torch wheel download"
  $PY -m pip install -q "dl/$W" 'numpy<2' 2>&1 | tail -3
  rm -f dl/$W
  done_ 3
fi

if step 4 "the rest of the python packages"; then
  $PY -m pip install -q biopython rdkit 'MDAnalysis<2.7' 'torch_geometric==2.3.1' 'e3nn==0.5.1' fair-esm PeptideBuilder opt-einsum 2>&1 | tail -5
  done_ 4
fi

say "torch check (before the CUDA ops are built):"
$PY - <<'PYEOF' 2>&1 | tail -8
import torch, time
print("torch", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    a = torch.randn(2048, 2048, device="cuda"); torch.cuda.synchronize(); t = time.time()
    for _ in range(10): b = a @ a
    torch.cuda.synchronize(); print("10 matmuls 2048^2: %.3f s" % (time.time() - t), "sum", float(b.sum()))
PYEOF

if step 5 "compile torch_scatter and torch_cluster for the Orin (sm_87) - this takes a while"; then
  export CUDA_HOME=/usr/local/cuda-11.4 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.7" MAX_JOBS=4
  export CPATH=$CUDAINC:$X/usr/include/aarch64-linux-gnu:${CPATH:-}
  export LIBRARY_PATH=$CUDALIB:$CUDNNLIB:${LIBRARY_PATH:-}
  export PATH=/usr/local/cuda-11.4/bin:$PATH
  $PY -m pip install --no-build-isolation --no-cache-dir torch-scatter==2.1.2 torch-cluster==1.6.3 > dl/build-ops.log 2>&1 || { tail -40 dl/build-ops.log; fail "building the CUDA ops"; }
  done_ 5
fi

say "FINAL CHECK"
$PY - <<'PYEOF' 2>&1 | tail -12
import torch, torch_scatter, torch_cluster
x = torch.randn(2000, 3, device="cuda")
e = torch_cluster.radius_graph(x, r=0.5)
k = torch_cluster.knn_graph(x, k=8)
s = torch_scatter.scatter(torch.ones(2000, device="cuda"), torch.randint(0, 10, (2000,), device="cuda"), reduce="sum")
print("radius_graph edges", e.shape[1], "knn", k.shape[1], "scatter", float(s.sum()))
import esm, e3nn, torch_geometric, rdkit, MDAnalysis, Bio, PeptideBuilder
print("imports ok: esm e3nn torch_geometric rdkit MDAnalysis")
PYEOF
say "ALL DONE"

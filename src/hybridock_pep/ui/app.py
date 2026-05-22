"""HybriDock-Pep Streamlit web UI.

Run with:  streamlit run src/hybridock_pep/ui/app.py
or via:    scripts/launch_ui.sh  (adds public URL via cloudflared)
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="HybriDock-Pep",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CAL = _REPO_ROOT / "data" / "calibration.json"

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Log fragment → (progress 0-1, display label)
_LOG_PROGRESS: list[tuple[str, float, str]] = [
    ("Stage 1: running RAPiDock", 0.05, "RAPiDock sampling (GPU)…"),
    ("Stage 1 bypassed", 0.30, "Poses loaded"),
    ("Stage 1 complete", 0.32, "Poses parsed"),
    ("Stage 1.5 complete", 0.40, "OpenMM minimization done"),
    ("Receptor prepared", 0.45, "Receptor PDBQT ready"),
    ("AD4 maps generated", 0.50, "AD4 grid maps ready"),
    ("Stage 2d", 0.58, "Vina + AD4 scoring…"),
    ("Stage 2 complete", 0.80, "Scoring complete"),
    ("Stage 3: clustering", 0.85, "Clustering poses…"),
    ("Stage 3 complete", 0.94, "Cluster analysis done"),
    ("Stage 4", 0.97, "Writing outputs…"),
]


# ── shared run context (thread-safe) ─────────────────────────────────────────

class RunContext:
    """Mutable container shared between the Streamlit main thread and pipeline thread.

    Session state stores a reference to this object; both threads share the same
    instance. All mutations go through the lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.logs: list[str] = []
        self.progress: float = 0.0
        self.stage: str = "Starting…"
        self.done: bool = False
        self.error: str | None = None
        # set on success
        self.scored_poses: list[Any] = []
        self.cluster_result: Any = None
        self.output_dir: Path | None = None

    def log(self, msg: str) -> None:
        with self._lock:
            self.logs.append(msg)

    def advance(self, value: float, stage: str) -> None:
        with self._lock:
            if value > self.progress:
                self.progress = value
                self.stage = stage

    def finish(self, scored_poses: list[Any], cluster_result: Any, output_dir: Path) -> None:
        with self._lock:
            self.scored_poses = scored_poses
            self.cluster_result = cluster_result
            self.output_dir = output_dir
            self.progress = 1.0
            self.stage = "Complete"
            self.done = True

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.done = True

    # Read-side helpers (snapshot under lock to avoid torn reads)
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "logs": list(self.logs),
                "progress": self.progress,
                "stage": self.stage,
                "done": self.done,
                "error": self.error,
                "scored_poses": list(self.scored_poses),
                "cluster_result": self.cluster_result,
                "output_dir": self.output_dir,
            }


class _UILogHandler(logging.Handler):
    """Routes log records from the hybridock_pep hierarchy into a RunContext."""

    def __init__(self, ctx: RunContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.ctx.log(msg)
        text = record.getMessage()
        for fragment, value, label in _LOG_PROGRESS:
            if fragment in text:
                self.ctx.advance(value, label)
                break


# ── pipeline thread ───────────────────────────────────────────────────────────

def _run_pipeline(
    ctx: RunContext,
    receptor_path: Path,
    peptide: str,
    site: tuple[float, float, float],
    box: float,
    n_samples: int,
    scoring_backends: set[str],
    output_dir: Path,
    input_poses_dir: Path | None,
    calibration_path: Path,
) -> None:
    """Entry point for the background pipeline thread. Never raises — errors go to ctx."""
    handler = _UILogHandler(ctx)

    # Attach only to hybridock_pep — all pipeline modules are under this hierarchy.
    # Do NOT also add to root: hybridock_pep propagates to root by default, so a
    # root attachment causes every record to hit the handler twice.
    hd_logger = logging.getLogger("hybridock_pep")
    hd_logger.addHandler(handler)
    old_hd_level = hd_logger.level
    hd_logger.setLevel(logging.DEBUG)

    try:
        from hybridock_pep.models import DockConfig
        from hybridock_pep.driver import run_dock

        config = DockConfig(
            peptide_sequence=peptide,
            receptor_path=receptor_path,
            site_coords=site,
            box_size=box,
            n_samples=n_samples,
            scoring=scoring_backends,  # type: ignore[arg-type]
            output_dir=output_dir,
        )

        scored_poses, cluster_result = run_dock(config, input_poses_dir, calibration_path)
        ctx.finish(scored_poses, cluster_result, output_dir)

    except Exception as exc:  # noqa: BLE001
        import traceback
        ctx.log(f"ERROR: {traceback.format_exc()}")
        ctx.fail(f"{type(exc).__name__}: {exc}")
    finally:
        hd_logger.removeHandler(handler)
        hd_logger.setLevel(old_hd_level)
        # Persist logs to disk so they survive session reconnects
        try:
            log_file = output_dir / "run.log"
            log_file.write_text("\n".join(ctx.snapshot()["logs"]))
        except Exception:  # noqa: BLE001
            pass


# ── GPU stats ─────────────────────────────────────────────────────────────────

def _gpu_stats() -> dict[str, str] | None:
    """Return GPU utilization / memory via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        name, util, mem_used, mem_total, temp = out.stdout.strip().split(",")
        return {
            "name": name.strip(),
            "util": f"{util.strip()} %",
            "memory": f"{mem_used.strip()} / {mem_total.strip()} MiB",
            "temp": f"{temp.strip()} °C",
        }
    except Exception:  # noqa: BLE001
        return None


# ── 3D viewer HTML ────────────────────────────────────────────────────────────

def _make_3dmol_html(receptor_pdb: str, peptide_pdb: str) -> str:
    """Return a self-contained HTML snippet embedding a 3Dmol.js viewer."""

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("`", r"\`").replace("$", r"\$")

    return f"""
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<div id="hd_viewer" style="
    width:100%; height:520px;
    position:relative;
    background:#f8f9fa;
    border-radius:8px;
    border:1px solid #e0e0e0;
"></div>
<script>
(function() {{
  let v = $3Dmol.createViewer(document.getElementById('hd_viewer'), {{
    backgroundColor: '#f8f9fa'
  }});
  v.addModel(`{_esc(receptor_pdb)}`, 'pdb');
  v.setStyle({{model:0}}, {{cartoon: {{color:'#7EB8DA', opacity:0.85}}}});
  v.addModel(`{_esc(peptide_pdb)}`, 'pdb');
  v.setStyle({{model:1}}, {{stick: {{colorscheme:'orangeCarbon', radius:0.25}}}});
  v.addSurface($3Dmol.SurfaceType.VDW, {{
    opacity:0.12, color:'#7EB8DA'
  }}, {{model:0}});
  v.zoomTo({{model:1}});
  v.render();
  v.spin(false);
}})();
</script>
<div style="font-size:11px;color:#888;margin-top:4px;">
  Receptor: blue cartoon &nbsp;|&nbsp; Peptide: orange sticks &nbsp;|&nbsp;
  Click-drag to rotate &nbsp;|&nbsp; Scroll to zoom
</div>
"""


# ── results helpers ───────────────────────────────────────────────────────────

def _build_df(scored_poses: list[Any]) -> pd.DataFrame:
    rows = []
    for p in scored_poses:
        rows.append({
            "Pose": p.pose_idx,
            "Hybrid (kcal/mol)": round(p.hybrid_score, 3) if p.hybrid_score is not None else None,
            "Vina (kcal/mol)": round(p.vina_score, 3) if p.vina_score is not None else None,
            "AD4 (kcal/mol)": round(p.ad4_score, 3) if p.ad4_score is not None else None,
            "Cluster": p.cluster_id,
            "Contacts": p.n_contact_residues,
            "Clashed": "⚠" if p.is_clashed else "",
        })
    df = pd.DataFrame(rows)
    if "Hybrid (kcal/mol)" in df.columns:
        df = df.sort_values("Hybrid (kcal/mol)").reset_index(drop=True)
    return df


# ── validation ────────────────────────────────────────────────────────────────

def _validate_peptide(seq: str) -> str | None:
    """Return error string or None."""
    up = seq.upper().strip()
    if not up:
        return "Peptide sequence is empty."
    bad = set(up) - _VALID_AA
    if bad:
        return f"Non-standard amino acid characters: {sorted(bad)}"
    if len(up) < 3:
        return "Peptide must have ≥ 3 residues."
    return None


# ── session state boot ────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "run_ctx": None,          # RunContext | None
        "run_thread": None,       # threading.Thread | None
        "temp_dir": None,         # str — persists across reruns so uploaded files survive
        "run_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict[str, Any] | None:
    """Render sidebar inputs. Returns param dict when Run is clicked, else None."""
    with st.sidebar:
        st.markdown("## 🔬 HybriDock-Pep")
        st.caption("Hybrid peptide docking · RAPiDock + physics rescoring")
        st.divider()

        # ── Mode ──
        mode = st.radio(
            "Mode",
            ["Score uploaded poses (CPU, ~30 s)", "Full dock — RAPiDock + scoring (GPU, ~5 min)"],
            index=0,
            help="Score-only mode takes pre-generated PDB poses and runs Vina/AD4 + clustering. "
                 "Full dock also runs RAPiDock diffusion sampling (requires CUDA GPU).",
        )
        score_only = mode.startswith("Score")
        st.divider()

        # ── Peptide ──
        st.markdown("**Peptide sequence**")
        peptide = st.text_input(
            "Sequence (1-letter AA codes)",
            value="LISDAELEAIFEADC",
            max_chars=50,
            label_visibility="collapsed",
        )
        err = _validate_peptide(peptide)
        if err:
            st.error(err)

        st.divider()

        # ── Receptor ──
        st.markdown("**Receptor PDB**")
        receptor_file = st.file_uploader(
            "Upload receptor PDB", type=["pdb"], label_visibility="collapsed",
        )

        st.divider()

        # ── Poses (score-only mode) ──
        pose_files = None
        if score_only:
            st.markdown("**Pre-generated pose PDBs**")
            pose_files = st.file_uploader(
                "Upload pose PDB files", type=["pdb"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="Upload the pose_*.pdb files from a previous RAPiDock run.",
            )

        # ── Binding site ──
        st.markdown("**Binding site**")
        c1, c2, c3 = st.columns(3)
        site_x = c1.number_input("X", value=31.9, format="%.1f", label_visibility="visible")
        site_y = c2.number_input("Y", value=17.5, format="%.1f", label_visibility="visible")
        site_z = c3.number_input("Z", value=9.5, format="%.1f", label_visibility="visible")
        box = st.number_input("Box size (Å)", value=30.0, min_value=5.0, max_value=100.0, step=1.0)
        # Warn when box is too small to contain a full-length peptide backbone.
        # Rule of thumb: ~2 Å × n_residues minimum (CLAUDE.md RTX_DEBUG Fix I).
        pep_len = len(peptide.strip()) if peptide.strip() else 0
        min_recommended = max(25, pep_len * 2)
        if not err and pep_len and box < min_recommended:
            st.warning(
                f"{pep_len}-mer needs box ≥ {min_recommended} Å "
                f"(~2 Å × residues). Current: {box:.0f} Å — poses will fall outside grid."
            )

        st.divider()

        # ── Advanced ──
        with st.expander("Advanced options"):
            n_samples = st.number_input("N samples (full dock only)", value=100, min_value=10,
                                        max_value=500, step=10)
            use_ad4 = st.checkbox("Enable AD4 scoring", value=True,
                                  help="Requires autogrid4 on PATH. Disable on macOS ARM.")
            cal_path_str = st.text_input(
                "Calibration JSON",
                value=str(_DEFAULT_CAL),
                help="Path to calibration.json with alpha/beta/gamma coefficients.",
            )

        st.divider()

        # ── Run button ──
        is_running = (
            st.session_state.run_ctx is not None
            and not st.session_state.run_ctx.snapshot()["done"]
        )

        run_clicked = st.button(
            "⏳ Running…" if is_running else "▶  Run",
            type="primary",
            disabled=bool(err) or is_running,
            use_container_width=True,
        )

        if run_clicked and not err:
            if receptor_file is None:
                st.error("Upload a receptor PDB first.")
                return None
            if score_only and not pose_files:
                st.error("Upload at least one pose PDB for score-only mode.")
                return None

            scoring_backends: set[str] = {"vina", "ad4"} if use_ad4 else {"vina"}
            cal_path = Path(cal_path_str)
            if not cal_path.exists():
                st.error(f"Calibration file not found: {cal_path}")
                return None

            return {
                "peptide": peptide.upper().strip(),
                "receptor_bytes": receptor_file.read(),
                "receptor_name": receptor_file.name,
                "pose_files": pose_files,  # list[UploadedFile] | None
                "site": (float(site_x), float(site_y), float(site_z)),
                "box": float(box),
                "n_samples": int(n_samples),
                "scoring_backends": scoring_backends,
                "score_only": score_only,
                "cal_path": cal_path,
            }

    return None


# ── main panel tabs ───────────────────────────────────────────────────────────

def _render_pipeline_tab(snap: dict[str, Any], is_running: bool) -> None:
    """Progress bar + GPU stats."""
    progress = snap["progress"]
    stage = snap["stage"]

    st.progress(progress, text=stage)

    # Stage checklist
    reached = [frag for frag, val, _ in _LOG_PROGRESS if val <= progress]
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Pipeline stages**")
        stage_labels = [label for _, _, label in _LOG_PROGRESS]
        for i, (_, val, label) in enumerate(_LOG_PROGRESS):
            if val <= progress:
                st.markdown(f"✅ {label}")
            elif i == len(reached):
                st.markdown(f"⏳ {label}")
                break

    with col2:
        gpu = _gpu_stats()
        if gpu:
            st.markdown("**GPU**")
            st.metric("Utilization", gpu["util"])
            st.metric("VRAM", gpu["memory"])
            st.metric("Temp", gpu["temp"])
            st.caption(gpu["name"])
        elif is_running:
            st.info("No GPU detected (score-only mode runs on CPU)")


def _render_results_tab(snap: dict[str, Any]) -> None:
    """Ranked table + plots + download buttons."""
    scored_poses = snap["scored_poses"]
    output_dir: Path | None = snap["output_dir"]
    cluster_result = snap["cluster_result"]

    # Also try loading from disk if scored_poses is empty but output_dir exists
    if not scored_poses and output_dir:
        ranked_csv_path = output_dir / "ranked_poses.csv"
        if ranked_csv_path.exists():
            st.info("Results loaded from disk.")
            df = pd.read_csv(ranked_csv_path)
            st.dataframe(df, height=300, use_container_width=True)
            st.divider()
            _render_downloads(output_dir)
            return

    if not scored_poses:
        st.info("No results yet — run the pipeline first.")
        return

    try:
        df = _build_df(scored_poses)

        # ── summary metrics ──
        hybrid_col = "Hybrid (kcal/mol)"
        best = float(df[hybrid_col].min())
        n_clusters = cluster_result.k_optimal if cluster_result else "—"
        sil = f"{cluster_result.silhouette_score:.3f}" if cluster_result else "—"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best hybrid score", f"{best:.2f} kcal/mol")
        c2.metric("Poses scored", len(scored_poses))
        c3.metric("Clusters (k)", n_clusters)
        c4.metric("Silhouette", sil)

        st.divider()

        # ── ranked table — plain dataframe, no Styler (pandas 3.0 compat) ──
        st.markdown("**Ranked poses**")
        st.dataframe(
            df,
            height=min(400, 40 + 35 * len(df)),
            use_container_width=True,
            column_config={
                hybrid_col: st.column_config.NumberColumn(format="%.3f"),
                "Vina (kcal/mol)": st.column_config.NumberColumn(format="%.3f"),
                "AD4 (kcal/mol)": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        return

    st.divider()

    # ── plots ──
    if output_dir:
        conv = output_dir / "convergence_plot.png"
        sil_img = output_dir / "silhouette_plot.png"
        has_plots = conv.exists() or sil_img.exists()
        if has_plots:
            img_cols = st.columns(2)
            if conv.exists():
                img_cols[0].image(str(conv), caption="Convergence")
            if sil_img.exists():
                img_cols[1].image(str(sil_img), caption="Silhouette k-selection")
        else:
            st.caption("No convergence/silhouette plots — need ≥ 2 poses for clustering.")

    st.divider()
    if output_dir:
        _render_downloads(output_dir)


def _render_downloads(output_dir: Path) -> None:
    st.markdown("**Downloads**")
    dl1, dl2, dl3 = st.columns(3)
    ranked_csv = output_dir / "ranked_poses.csv"
    best_pdb = output_dir / "best_pose.pdb"
    cluster_csv = output_dir / "cluster_summary.csv"
    if ranked_csv.exists():
        dl1.download_button("ranked_poses.csv", ranked_csv.read_bytes(),
                            file_name="ranked_poses.csv", mime="text/csv")
    if best_pdb.exists():
        dl2.download_button("best_pose.pdb", best_pdb.read_bytes(),
                            file_name="best_pose.pdb", mime="chemical/x-pdb")
    if cluster_csv.exists():
        dl3.download_button("cluster_summary.csv", cluster_csv.read_bytes(),
                            file_name="cluster_summary.csv", mime="text/csv")


def _render_3d_tab(snap: dict[str, Any], receptor_path: Path | None) -> None:
    """3Dmol.js viewer: receptor + best pose."""
    output_dir: Path | None = snap["output_dir"]

    best_pdb_path = output_dir / "best_pose.pdb" if output_dir else None
    if best_pdb_path is None or not best_pdb_path.exists():
        st.info("Run the pipeline to see the best-pose 3D structure.")
        return
    if receptor_path is None or not receptor_path.exists():
        st.info("Receptor file not available for 3D view.")
        return

    receptor_pdb = receptor_path.read_text()
    peptide_pdb = best_pdb_path.read_text()

    st.markdown(
        "**Receptor** (blue cartoon) + **best-pose peptide** (orange sticks)"
    )
    html = _make_3dmol_html(receptor_pdb, peptide_pdb)
    st.components.v1.html(html, height=560, scrolling=False)


def _render_logs_tab(snap: dict[str, Any]) -> None:
    """Scrollable raw log stream."""
    logs = snap["logs"]
    output_dir: Path | None = snap["output_dir"]

    # Fall back to on-disk log file if session-state logs are empty
    if not logs and output_dir:
        log_file = output_dir / "run.log"
        if log_file.exists():
            logs = log_file.read_text().splitlines()

    if not logs:
        if snap["done"]:
            st.warning("No logs captured. The pipeline may have run in a previous session.")
        else:
            st.info("Logs will appear here once the pipeline starts.")
        return

    log_text = "\n".join(logs[-500:])
    st.code(log_text, language="text")


# ── main ──────────────────────────────────────────────────────────────────────

def app() -> None:
    _init_state()

    # Ensure a temp dir persists across Streamlit reruns for this session
    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="hybridock_")
    tmp = Path(st.session_state.temp_dir)

    # ── sidebar / run trigger ──────────────────────────────────────────────────
    params = _render_sidebar()

    if params is not None:
        # Save uploaded receptor to temp dir
        receptor_path = tmp / params["receptor_name"]
        receptor_path.write_bytes(params["receptor_bytes"])

        # Save pose PDBs if score-only mode
        input_poses_dir: Path | None = None
        if params["score_only"] and params["pose_files"]:
            poses_tmp = tmp / f"poses_{st.session_state.run_count}"
            poses_tmp.mkdir(exist_ok=True)
            for uf in params["pose_files"]:
                (poses_tmp / uf.name).write_bytes(uf.read())
            input_poses_dir = poses_tmp

        output_dir = tmp / f"run_{st.session_state.run_count}"
        output_dir.mkdir(exist_ok=True)
        st.session_state.run_count += 1

        # Create run context and launch thread
        ctx = RunContext()
        st.session_state.run_ctx = ctx

        thread = threading.Thread(
            target=_run_pipeline,
            kwargs={
                "ctx": ctx,
                "receptor_path": receptor_path,
                "peptide": params["peptide"],
                "site": params["site"],
                "box": params["box"],
                "n_samples": params["n_samples"],
                "scoring_backends": params["scoring_backends"],
                "output_dir": output_dir,
                "input_poses_dir": input_poses_dir,
                "calibration_path": params["cal_path"],
            },
            daemon=True,
        )
        thread.start()
        st.session_state.run_thread = thread
        st.rerun()

    # ── main panel ────────────────────────────────────────────────────────────
    st.title("🔬 HybriDock-Pep")
    st.caption("RAPiDock diffusion sampling · Vina + AD4 rescoring · Kabsch-RMSD clustering")

    ctx: RunContext | None = st.session_state.run_ctx
    snap: dict[str, Any] = ctx.snapshot() if ctx is not None else {
        "logs": [], "progress": 0.0, "stage": "Ready",
        "done": False, "error": None,
        "scored_poses": [], "cluster_result": None, "output_dir": None,
    }

    is_running = ctx is not None and not snap["done"]

    # Detect the running→done transition so we fire one final rerun that carries
    # a fresh snapshot with scored_poses populated.
    was_running = st.session_state.get("_was_running", False)
    just_finished = was_running and not is_running and snap["done"]
    st.session_state["_was_running"] = is_running

    # Error banner
    if snap["error"]:
        st.error(f"Pipeline failed: {snap['error']}")

    # Success banner
    if snap["done"] and not snap["error"] and snap["scored_poses"]:
        st.success(
            f"Pipeline complete — {len(snap['scored_poses'])} poses scored, "
            f"best hybrid score: {min(p.hybrid_score for p in snap['scored_poses']):.2f} kcal/mol"
        )

    # Running banner
    if is_running:
        st.info(f"⏳ {snap['stage']}  ({snap['progress']*100:.0f}%)")

    # Derive receptor_path for 3D viewer (last receptor uploaded to temp dir)
    receptor_path: Path | None = None
    for f in tmp.glob("*.pdb"):
        if "pose" not in f.name and "best" not in f.name:
            receptor_path = f
            break

    # ── tabs ──
    tab_pipeline, tab_results, tab_3d, tab_logs = st.tabs([
        "📊 Pipeline", "📋 Results", "🧬 3D View", "📜 Logs"
    ])

    with tab_pipeline:
        _render_pipeline_tab(snap, is_running)

    with tab_results:
        _render_results_tab(snap)

    with tab_3d:
        _render_3d_tab(snap, receptor_path)

    with tab_logs:
        _render_logs_tab(snap)

    # ── auto-rerun while pipeline is running ──────────────────────────────────
    if is_running:
        time.sleep(0.8)
        st.rerun()
    elif just_finished:
        # One guaranteed final rerun so the page renders with the completed snapshot.
        st.rerun()


if __name__ == "__main__":
    app()
else:
    # Streamlit imports the module; call app() at module level
    app()

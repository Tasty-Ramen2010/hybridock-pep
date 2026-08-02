"""Does the receptor-prep backend change the answer?

Dropping ADFRsuite means receptor PDBQTs now come from meeko (or, as a last
resort, obabel) instead of ADFRsuite's ``prepare_receptor``. The obvious worry
is that a different backend assigns different partial charges and atom types,
and that the shipped calibration — fitted back when ADFRsuite was the only
path — silently stops applying.

These tests pin the measurements that resolve that worry, so the reasoning in
``prep/receptor.py`` and ``hybridock-pep guide prep`` cannot quietly rot:

  1. Vina ignores the receptor charge column outright. Charges are the property
     backends disagree on most, and they reach nothing on the default path.
  2. Vina *does* read the atom-type column — tested explicitly, so (1) can
     never pass vacuously because the harness stopped coupling receptor to
     ligand.
  3. The specific type substitutions meeko and obabel disagree on (N/NA, A/C,
     S/SA) are each exactly score-neutral in Vina. Only O/OA moves a score, and
     the backends agree on every oxygen.

Together these say the backend cannot move a Vina score on this system, so it
cannot move the reported ΔG either — Vina only ever perturbs the answer through
clash relief, which acts on the score.

The Vina-scoring tests are marked ``slow`` (grid-map construction dominates);
run them with ``pytest -m slow``. The parsing tests are cheap and always run.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
RECEPTOR_PDB = REPO / "data" / "pdbs" / "1YCR_mdm2.pdb"
PEPTIDE_PDB = REPO / "data" / "pdbs" / "1YCR_peptide.pdb"

requires_meeko = pytest.mark.skipif(
    shutil.which("mk_prepare_receptor.py") is None, reason="meeko not installed"
)
requires_obabel = pytest.mark.skipif(
    shutil.which("obabel") is None and shutil.which("babel") is None,
    reason="openbabel not installed",
)
requires_structures = pytest.mark.skipif(
    not (RECEPTOR_PDB.is_file() and PEPTIDE_PDB.is_file()),
    reason="1YCR reference structures not present",
)


def _vina():
    return pytest.importorskip("vina", reason="AutoDock Vina python bindings not installed")


# ---------------------------------------------------------------------------
# Fixtures: build each receptor PDBQT once per session (meeko/obabel are slow)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def meeko_receptor(tmp_path_factory) -> Path:
    from hybridock_pep.models import DockConfig
    from hybridock_pep.prep.receptor import prepare_receptor

    out = tmp_path_factory.mktemp("meeko_rec")
    cfg = DockConfig(
        peptide_sequence="ETFSDLWKLLPE",
        receptor_path=RECEPTOR_PDB,
        output_dir=out,
        # The real 1YCR pocket. Not used for PDBQT generation, but wrong
        # coordinates here would silently break any grid-map work added later.
        site_coords=(25.20, -25.61, -7.97),
        box_size=30.0,
    )
    return prepare_receptor(cfg)


@pytest.fixture(scope="session")
def obabel_receptor(tmp_path_factory) -> Path:
    from hybridock_pep.prep.pdbqt_convert import convert_pdb_to_pdbqt

    out = tmp_path_factory.mktemp("obabel_rec") / "receptor.pdbqt"
    result = convert_pdb_to_pdbqt(RECEPTOR_PDB, out, add_hydrogens=True)
    if result.returncode != 0 or not out.is_file():
        pytest.skip("obabel receptor conversion failed")
    return out


@pytest.fixture(scope="session")
def ligand_pdbqt(tmp_path_factory) -> Path:
    from hybridock_pep.prep.ligand import _prepare_single_ligand

    out = tmp_path_factory.mktemp("lig")
    res = _prepare_single_ligand((0, PEPTIDE_PDB, out))
    if not isinstance(res, Path) or not res.is_file():
        pytest.skip(f"ligand PDBQT preparation failed: {res}")
    return res


# ---------------------------------------------------------------------------
# PDBQT helpers
# ---------------------------------------------------------------------------

HYDROGEN_TYPES = {"H", "HD"}


def atom_types(pdbqt: Path, heavy_only: bool = True) -> dict:
    """Map rounded (x, y, z) -> AutoDock atom type.

    Keyed on coordinates rather than residue names on purpose: obabel
    renumbers and renames residues, so a name-based key matches almost nothing
    and would make a comparison look artificially clean.
    """
    out = {}
    for line in pdbqt.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        t = line[77:79].strip()
        if heavy_only and t in HYDROGEN_TYPES:
            continue
        key = (
            round(float(line[30:38]), 2),
            round(float(line[38:46]), 2),
            round(float(line[46:54]), 2),
        )
        out[key] = t
    return out


def rewrite(src: Path, dst: Path, *, charge=None, atom_type=None) -> Path:
    """Copy a PDBQT, transforming the charge and/or type column only.

    Coordinates are never touched, so any score change is attributable purely
    to the column under test.
    """
    lines = []
    for line in src.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 79:
            q = float(line[70:76])
            t = line[77:79].strip()
            new_q = charge(q) if charge else q
            new_t = atom_type(t) if atom_type else t
            line = f"{line[:70]}{new_q:6.3f}{line[76:77]}{new_t:<2s}{line[79:]}"
        lines.append(line)
    dst.write_text("".join(lines))
    return dst


def vina_score(receptor: Path, ligand: Path) -> float:
    """Vina intermolecular score, at full precision.

    ``Vina.score()`` rounds to 3 decimals, which is coarse enough to hide a
    small dependence and turn a sensitivity test into a false negative. Read
    the raw C++ value instead.
    """
    from vina import Vina

    xs, ys, zs = [], [], []
    for line in ligand.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
    center = [sum(a) / len(a) for a in (xs, ys, zs)]
    box = [max(22.0, max(a) - min(a) + 10.0) for a in (xs, ys, zs)]

    v = Vina(sf_name="vina", seed=42, verbosity=0)
    v.set_receptor(str(receptor))
    v.set_ligand_from_file(str(ligand))
    v.compute_vina_maps(center=center, box_size=box)
    return float(v._vina.score()[0])


# ---------------------------------------------------------------------------
# What meeko produces
# ---------------------------------------------------------------------------

@requires_structures
@requires_meeko
class TestMeekoOutput:
    def test_emits_canonical_autodock_types(self, meeko_receptor):
        """Only real AutoDock types, and the polar ones must be present.

        A backend that silently degraded to bare elements (no OA/HD/NA/SA)
        would still produce a loadable PDBQT but a chemically wrong one.
        """
        types = set(atom_types(meeko_receptor, heavy_only=False).values())
        assert types <= {"C", "A", "N", "NA", "OA", "O", "SA", "S", "HD", "H", "P", "F", "Cl", "Br", "I"}, (
            f"unexpected atom types: {sorted(types)}"
        )
        for required in ("OA", "HD", "N", "C"):
            assert required in types, f"meeko produced no {required} atoms"

    def test_merges_nonpolar_hydrogens(self, meeko_receptor):
        """AutoDock is a united-atom model: only polar H (HD) survive.

        If plain 'H' shows up, nonpolar hydrogens were not merged and every
        Vina score would be computed against the wrong sterics.
        """
        all_types = list(atom_types(meeko_receptor, heavy_only=False).values())
        assert "H" not in all_types, "nonpolar hydrogens were not merged"
        assert all_types.count("HD") > 0

    def test_backbone_amide_nitrogen_is_donor_not_acceptor(self, meeko_receptor):
        """N (donor) rather than NA (acceptor) is the chemically correct call.

        This is the single largest disagreement with the obabel fallback, and
        meeko is the one that gets it right.
        """
        counts = {}
        for t in atom_types(meeko_receptor).values():
            counts[t] = counts.get(t, 0) + 1
        assert counts.get("N", 0) > counts.get("NA", 0), (
            f"expected mostly donor N, got N={counts.get('N', 0)} NA={counts.get('NA', 0)}"
        )


# ---------------------------------------------------------------------------
# Charges: provably irrelevant to Vina
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_structures
@requires_meeko
class TestVinaIgnoresReceptorCharges:
    def test_zeroing_all_charges_changes_nothing(self, meeko_receptor, ligand_pdbqt, tmp_path):
        _vina()
        base = vina_score(meeko_receptor, ligand_pdbqt)
        zeroed = rewrite(meeko_receptor, tmp_path / "zero.pdbqt", charge=lambda q: 0.0)
        assert vina_score(zeroed, ligand_pdbqt) == pytest.approx(base, abs=1e-9)

    def test_absurd_charges_change_nothing(self, meeko_receptor, ligand_pdbqt, tmp_path):
        """Sign-flipped and tripled — physically impossible, still identical.

        This is what makes backend charge-assignment differences a non-issue on
        the default --scoring vina path.
        """
        _vina()
        base = vina_score(meeko_receptor, ligand_pdbqt)
        absurd = rewrite(meeko_receptor, tmp_path / "absurd.pdbqt", charge=lambda q: -3.0 * q)
        assert vina_score(absurd, ligand_pdbqt) == pytest.approx(base, abs=1e-9)


# ---------------------------------------------------------------------------
# Types: Vina does read them — so the charge tests above aren't vacuous
# ---------------------------------------------------------------------------

@pytest.mark.slow
@requires_structures
@requires_meeko
class TestVinaDoesReadAtomTypes:
    def test_flattening_every_heavy_atom_to_carbon_moves_the_score(
        self, meeko_receptor, ligand_pdbqt, tmp_path
    ):
        """Control for the charge tests.

        If this passed as 'no change' too, the harness would not be coupling
        receptor to ligand at all and every other result here would be void.
        """
        _vina()
        base = vina_score(meeko_receptor, ligand_pdbqt)
        flat = rewrite(
            meeko_receptor, tmp_path / "flat.pdbqt",
            atom_type=lambda t: t if t in HYDROGEN_TYPES else "C",
        )
        assert abs(vina_score(flat, ligand_pdbqt) - base) > 0.5, (
            "retyping every heavy atom to carbon did not move the Vina score — "
            "the receptor is not reaching the scoring function"
        )

    def test_stripping_oxygen_acceptor_flag_moves_the_score(
        self, meeko_receptor, ligand_pdbqt, tmp_path
    ):
        """O/OA is the one substitution that matters.

        Documents *why* meeko-vs-obabel oxygen agreement is load-bearing.
        """
        _vina()
        base = vina_score(meeko_receptor, ligand_pdbqt)
        stripped = rewrite(
            meeko_receptor, tmp_path / "no_oa.pdbqt",
            atom_type=lambda t: "O" if t == "OA" else t,
        )
        assert abs(vina_score(stripped, ligand_pdbqt) - base) > 0.01


@pytest.mark.slow
@requires_structures
@requires_meeko
@pytest.mark.parametrize(
    "label,fn",
    [
        ("N -> NA", lambda t: "NA" if t == "N" else t),
        ("NA -> N", lambda t: "N" if t == "NA" else t),
        ("A -> C", lambda t: "C" if t == "A" else t),
        ("SA -> S", lambda t: "S" if t == "SA" else t),
    ],
)
def test_backend_disagreement_types_are_score_neutral(
    meeko_receptor, ligand_pdbqt, tmp_path, label, fn
):
    """Every type pair meeko and obabel disagree on is neutral in Vina.

    Vina collapses N/NA (donor status comes from attached HD), has a single
    carbon type (aromatic vs aliphatic is irrelevant), and gives sulfur no
    hydrogen-bond term. So all three disagreement classes cancel.
    """
    _vina()
    base = vina_score(meeko_receptor, ligand_pdbqt)
    variant = rewrite(meeko_receptor, tmp_path / "v.pdbqt", atom_type=fn)
    assert vina_score(variant, ligand_pdbqt) == pytest.approx(base, abs=1e-9), (
        f"{label} was expected to be score-neutral in Vina but moved the score"
    )


# ---------------------------------------------------------------------------
# The fallback chain must actually be a chain
# ---------------------------------------------------------------------------

@requires_structures
class TestMeekoFailureFallsThroughToObabel:
    """A meeko failure must degrade to obabel, never abort receptor prep.

    meeko matches every residue against a template library and refuses the
    whole structure if any residue has an unexpected heavy-atom set — real
    crystal receptors hit this routinely (``No template matched for
    residue_key='A:41' ... PHE heavy_miss=0 heavy_excess=1``). Two of the
    bundled e2e fixtures, kin_2khh and mdm2_1pmx, fail exactly this way while
    obabel handles both.

    Raising instead of falling through silently reduced the documented
    ``prepare_receptor -> mk_prepare_receptor.py -> obabel`` chain to two
    steps and broke every receptor meeko could not template-match.
    """

    def test_meeko_helper_returns_false_instead_of_raising(self, tmp_path, monkeypatch):
        import subprocess as sp

        from hybridock_pep.prep import receptor as rec_mod

        def fake_run(cmd, **kw):
            return sp.CompletedProcess(cmd, 1, stdout="", stderr="No template matched")

        monkeypatch.setattr(rec_mod.subprocess, "run", fake_run)
        out = tmp_path / "receptor.pdbqt"
        assert rec_mod._try_meeko(tmp_path / "in.pdb", out, "/opt/env/bin/mk_prepare_receptor.py") is False
        assert not out.exists(), "a failed meeko run must not leave a partial file"

    def test_meeko_helper_reports_success(self, tmp_path, monkeypatch):
        import subprocess as sp

        from hybridock_pep.prep import receptor as rec_mod

        out = tmp_path / "receptor.pdbqt"

        def fake_run(cmd, **kw):
            out.write_text("ATOM\n")
            return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(rec_mod.subprocess, "run", fake_run)
        assert rec_mod._try_meeko(tmp_path / "in.pdb", out, "/opt/env/bin/mk_prepare_receptor.py") is True

    def test_partial_output_with_zero_exit_is_still_a_failure(self, tmp_path, monkeypatch):
        """meeko can exit 0 having written nothing (D-02)."""
        import subprocess as sp

        from hybridock_pep.prep import receptor as rec_mod

        monkeypatch.setattr(
            rec_mod.subprocess, "run",
            lambda cmd, **kw: sp.CompletedProcess(cmd, 0, stdout="", stderr=""),
        )
        assert rec_mod._try_meeko(tmp_path / "in.pdb", tmp_path / "receptor.pdbqt", "/opt/env/bin/mk_prepare_receptor.py") is False

    @requires_obabel
    @pytest.mark.parametrize("fixture", ["kin_2khh", "mdm2_1pmx"])
    def test_receptors_meeko_rejects_still_prepare(self, tmp_path, fixture):
        """End to end on the two fixtures that actually trip meeko."""
        from hybridock_pep.models import DockConfig
        from hybridock_pep.prep.receptor import prepare_receptor

        src = REPO / "tests" / "fixtures" / fixture / "receptor_pocket.pdb"
        if not src.is_file():
            pytest.skip(f"fixture {fixture} not present")
        cfg = DockConfig(
            peptide_sequence="ACDEFGHIK",
            receptor_path=src,
            output_dir=tmp_path,
            site_coords=(0.0, 0.0, 0.0),
            box_size=30.0,
        )
        out = prepare_receptor(cfg)
        assert out.is_file() and out.stat().st_size > 0, (
            f"{fixture} produced no receptor PDBQT — the meeko->obabel fallback "
            "is not working"
        )


# ---------------------------------------------------------------------------
# meeko vs the obabel fallback, end to end
# ---------------------------------------------------------------------------

@requires_structures
@requires_meeko
@requires_obabel
class TestMeekoVersusObabel:
    def test_same_heavy_atom_set(self, meeko_receptor, obabel_receptor):
        m, o = atom_types(meeko_receptor), atom_types(obabel_receptor)
        assert set(m) == set(o), (
            f"heavy-atom coordinates differ: meeko {len(m)}, obabel {len(o)}, "
            f"shared {len(set(m) & set(o))}"
        )

    def test_they_agree_on_every_oxygen(self, meeko_receptor, obabel_receptor):
        """O/OA is the only substitution Vina reacts to, so this is the one
        agreement the equal-score result actually depends on."""
        m, o = atom_types(meeko_receptor), atom_types(obabel_receptor)
        mismatched = [
            k for k in set(m) & set(o)
            if {m[k], o[k]} & {"O", "OA"} and m[k] != o[k]
        ]
        assert not mismatched, f"{len(mismatched)} oxygen atoms typed differently"

    def test_disagreements_are_confined_to_the_neutral_classes(
        self, meeko_receptor, obabel_receptor
    ):
        neutral = ({"N", "NA"}, {"A", "C"}, {"S", "SA"})
        m, o = atom_types(meeko_receptor), atom_types(obabel_receptor)
        unexpected = []
        for k in set(m) & set(o):
            if m[k] == o[k]:
                continue
            if not any({m[k], o[k]} <= pair for pair in neutral):
                unexpected.append((m[k], o[k]))
        assert not unexpected, f"disagreements outside the neutral classes: {set(unexpected)}"

    @pytest.mark.slow
    def test_produce_the_same_vina_score(self, meeko_receptor, obabel_receptor, ligand_pdbqt):
        """The bottom line: swapping the receptor-prep backend does not move
        the number the pipeline reports."""
        _vina()
        assert vina_score(meeko_receptor, ligand_pdbqt) == pytest.approx(
            vina_score(obabel_receptor, ligand_pdbqt), abs=1e-6
        )

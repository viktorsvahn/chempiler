"""Tests for the sphere-mode perception module (chempiler.sphere)."""

import numpy as np
import pytest
from ase import Atoms

from chempiler.sphere import (
    parse_cutoffs,
    detect_coordination_centers,
    detect_sphere_cutoffs,
    build_molecules_sphere,
    _COVALENT_ELEMENTS,
)


# ============================================================
# helpers
# ============================================================

def be_water_atoms(be_o_dist=2.0, cell=15.0):
    """Be at cell centre with one H2O at be_o_dist."""
    c = cell / 2
    return Atoms(
        symbols=["Be", "O", "H", "H"],
        positions=[
            [c, c, c],
            [c + be_o_dist, c, c],
            [c + be_o_dist + 0.96, c, c],
            [c + be_o_dist, c + 0.96, c],
        ],
        cell=[cell] * 3,
        pbc=True,
    )


def two_water_atoms(cell=15.0):
    """Two H2O molecules — no metal centre, pure covalent elements."""
    return Atoms(
        symbols=["O", "H", "H", "O", "H", "H"],
        positions=[
            [1.0, 0, 0], [1.96, 0, 0], [1.0, 0.96, 0],
            [5.0, 0, 0], [5.96, 0, 0], [5.0, 0.96, 0],
        ],
        cell=[cell] * 3,
        pbc=True,
    )


def hbond_water_atoms():
    """Two H2O with an H-bond at ~1.84 Å — should remain separate molecules."""
    return Atoms(
        symbols=["O", "H", "H", "O", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0],
            [2.8, 0.0, 0.0], [3.76, 0.0, 0.0], [2.8, 0.96, 0.0],
        ],
        cell=[15.0] * 3,
        pbc=True,
    )


def be_coordination_frame(r_first=2.0, n_ligands=4, cell=20.0):
    """Be at centre with n_ligands O atoms in the first shell."""
    c = cell / 2
    symbols = ["Be"]
    positions = [[c, c, c]]
    angles = np.linspace(0, 2 * np.pi, n_ligands, endpoint=False)
    for a in angles:
        symbols.append("O")
        positions.append([c + r_first * np.cos(a), c + r_first * np.sin(a), c])
    return Atoms(symbols=symbols, positions=positions, cell=[cell] * 3, pbc=True)


# ============================================================
# parse_cutoffs
# ============================================================

class TestParseCutoffs:

    def test_none_returns_empty(self):
        assert parse_cutoffs(None) == {}

    def test_empty_dict_returns_empty(self):
        assert parse_cutoffs({}) == {}

    def test_string_key_parsed(self):
        result = parse_cutoffs({"Be-O": 2.4})
        assert ("Be", "O") in result
        assert abs(result[("Be", "O")] - 2.4) < 1e-10

    def test_tuple_key_preserved(self):
        result = parse_cutoffs({("Be", "O"): 2.4})
        assert ("Be", "O") in result

    def test_value_is_float(self):
        result = parse_cutoffs({"Be-O": 2})
        assert isinstance(result[("Be", "O")], float)

    def test_multiple_pairs(self):
        result = parse_cutoffs({"Be-O": 2.4, "Be-Cl": 2.8})
        assert len(result) == 2

    def test_invalid_string_key_raises(self):
        with pytest.raises(ValueError):
            parse_cutoffs({"BeO": 2.4})

    def test_whitespace_in_key_stripped(self):
        result = parse_cutoffs({"Be - O": 2.4})
        assert ("Be", "O") in result


# ============================================================
# detect_coordination_centers
# ============================================================

class TestDetectCoordinationCenters:

    def test_pure_water_returns_empty(self):
        atoms_list = [two_water_atoms()]
        centers = detect_coordination_centers(atoms_list)
        assert len(centers) == 0

    def test_be_detected_as_center(self):
        atoms_list = [be_water_atoms()]
        centers = detect_coordination_centers(atoms_list)
        assert "Be" in centers

    def test_h_o_not_in_centers(self):
        atoms_list = [be_water_atoms()]
        centers = detect_coordination_centers(atoms_list)
        assert "H" not in centers
        assert "O" not in centers

    def test_all_covalent_elements_excluded(self):
        for elem in _COVALENT_ELEMENTS:
            assert elem not in detect_coordination_centers(
                [Atoms(symbols=[elem], positions=[[0, 0, 0]], cell=[10]*3, pbc=True)]
            )

    def test_multiple_frames_combined(self):
        f1 = Atoms(symbols=["Na"], positions=[[0, 0, 0]], cell=[10]*3, pbc=True)
        f2 = Atoms(symbols=["Ca"], positions=[[0, 0, 0]], cell=[10]*3, pbc=True)
        centers = detect_coordination_centers([f1, f2])
        assert "Na" in centers
        assert "Ca" in centers


# ============================================================
# detect_sphere_cutoffs
# ============================================================

class TestDetectSphereCutoffs:

    def test_first_shell_cutoff_detected(self):
        """Be with 6 O at 2.0 Å — cutoff should be detected between 2 and 4 Å."""
        frame = be_coordination_frame(r_first=2.0, n_ligands=6, cell=20.0)
        atoms_list = [frame] * 50
        centers = {"Be"}
        cutoffs = detect_sphere_cutoffs(atoms_list, centers, ligands=("O",))
        assert ("Be", "O") in cutoffs
        # The cutoff lies near the first minimum after the first shell
        # (~2 Å); with histogram discretisation it may be just below 2.0 Å.
        assert 1.5 < cutoffs[("Be", "O")] < 4.5

    def test_absent_pair_not_in_result(self):
        """If no N atoms in the frames, Be-N should not appear."""
        frame = be_coordination_frame(r_first=2.0, n_ligands=4, cell=20.0)
        cutoffs = detect_sphere_cutoffs([frame] * 20, {"Be"}, ligands=("N",))
        assert len(cutoffs) == 0

    def test_returns_dict(self):
        frame = be_coordination_frame()
        result = detect_sphere_cutoffs([frame] * 10, {"Be"}, ligands=("O",))
        assert isinstance(result, dict)

    def test_cutoff_is_float(self):
        frame = be_coordination_frame()
        result = detect_sphere_cutoffs([frame] * 50, {"Be"}, ligands=("O",))
        for v in result.values():
            assert isinstance(v, float)


# ============================================================
# build_molecules_sphere
# ============================================================

class TestBuildMoleculesSphere:

    def test_returns_list_of_lists(self):
        atoms = two_water_atoms()
        mols = build_molecules_sphere(atoms, {})
        assert isinstance(mols, list)
        assert all(isinstance(m, list) for m in mols)

    def test_all_atoms_covered(self):
        atoms = be_water_atoms()
        sphere_cutoffs = {("Be", "O"): 2.5}
        mols = build_molecules_sphere(atoms, sphere_cutoffs)
        covered = {a for mol in mols for a in mol}
        assert covered == set(range(len(atoms)))

    def test_each_atom_in_one_molecule(self):
        atoms = be_water_atoms()
        sphere_cutoffs = {("Be", "O"): 2.5}
        mols = build_molecules_sphere(atoms, sphere_cutoffs)
        flat = [a for mol in mols for a in mol]
        assert len(flat) == len(set(flat))

    def test_be_merges_with_nearby_water(self):
        """Be at 2.0 Å from O with cutoff 2.5 Å → all 4 atoms in one group."""
        atoms = be_water_atoms(be_o_dist=2.0)
        sphere_cutoffs = {("Be", "O"): 2.5}
        mols = build_molecules_sphere(atoms, sphere_cutoffs)
        # All atoms should be in one group
        assert len(mols) == 1
        assert set(mols[0]) == {0, 1, 2, 3}

    def test_be_outside_cutoff_stays_isolated(self):
        """Be at 4.0 Å from O with cutoff 2.5 Å → Be alone, H2O separate."""
        atoms = be_water_atoms(be_o_dist=4.0)
        sphere_cutoffs = {("Be", "O"): 2.5}
        mols = build_molecules_sphere(atoms, sphere_cutoffs)
        assert len(mols) == 2
        be_group = next(m for m in mols if 0 in m)
        assert be_group == [0]

    def test_hbond_not_treated_as_covalent(self):
        """H···O at 1.84 Å (H-bond) must not create a bond in sphere mode."""
        atoms = hbond_water_atoms()
        # No sphere cutoffs → pure tight covalent
        mols = build_molecules_sphere(atoms, {})
        assert len(mols) == 2

    def test_empty_cutoffs_gives_covalent_result(self):
        """Empty sphere_cutoffs → tight covalent only; normal H2O bonds captured."""
        atoms = two_water_atoms()
        mols = build_molecules_sphere(atoms, {})
        assert len(mols) == 2
        sizes = sorted(len(m) for m in mols)
        assert sizes == [3, 3]

    def test_bond_scale_reduces_covalent_cutoffs(self):
        """bond_scale=0 → no covalent bonds → each atom isolated (except centres)."""
        atoms = two_water_atoms()
        mols = build_molecules_sphere(atoms, {}, bond_scale=0.0)
        assert len(mols) == 6  # 6 separate atoms

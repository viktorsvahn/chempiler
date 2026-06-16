"""Tests for UnionFind and build_molecules (chempiler.perception)."""

import numpy as np
import pytest
from ase import Atoms

from chempiler.perception import UnionFind, build_molecules


# ============================================================
# UnionFind
# ============================================================

class TestUnionFind:

    def test_initial_find_is_self(self):
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_merges_sets(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_transitivity(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_disjoint_sets_stay_separate(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) != uf.find(2)

    def test_union_self_is_noop(self):
        uf = UnionFind(3)
        uf.union(1, 1)
        assert uf.find(1) == uf.find(1)

    def test_union_already_connected_noop(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        root_before = uf.find(0)
        uf.union(0, 1)
        assert uf.find(0) == root_before

    def test_all_in_one_set(self):
        n = 6
        uf = UnionFind(n)
        for i in range(n - 1):
            uf.union(i, i + 1)
        roots = {uf.find(i) for i in range(n)}
        assert len(roots) == 1

    def test_n_equals_one(self):
        uf = UnionFind(1)
        assert uf.find(0) == 0

    def test_path_compression_consistency(self):
        # After path compression, find() must still return the same root
        uf = UnionFind(5)
        for i in range(4):
            uf.union(i, i + 1)
        root = uf.find(4)
        for _ in range(10):
            assert uf.find(4) == root


# ============================================================
# build_molecules
# ============================================================

def water_atoms(cell=20.0):
    """One H2O molecule well inside a large cell."""
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[5.0, 5.0, 5.0], [5.96, 5.0, 5.0], [5.0, 5.96, 5.0]],
        cell=[cell] * 3,
        pbc=True,
    )


def two_water_atoms():
    """Two H2O molecules 8 Å apart — should not be merged in molecular mode."""
    return Atoms(
        symbols=["O", "H", "H", "O", "H", "H"],
        positions=[
            [5.0, 5.0, 5.0], [5.96, 5.0, 5.0], [5.0, 5.96, 5.0],
            [13.0, 5.0, 5.0], [13.96, 5.0, 5.0], [13.0, 5.96, 5.0],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )


def isolated_atoms():
    """Four separate atoms, each 10 Å apart — no bonds possible."""
    return Atoms(
        symbols=["O", "N", "C", "H"],
        positions=[[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]],
        cell=[50.0, 50.0, 50.0],
        pbc=True,
    )


class TestBuildMolecules:

    def test_returns_list_of_lists(self):
        mols = build_molecules(water_atoms())
        assert isinstance(mols, list)
        assert all(isinstance(m, list) for m in mols)

    def test_all_atoms_covered(self):
        atoms = two_water_atoms()
        mols = build_molecules(atoms)
        covered = {a for mol in mols for a in mol}
        assert covered == set(range(len(atoms)))

    def test_each_atom_in_exactly_one_molecule(self):
        atoms = two_water_atoms()
        mols = build_molecules(atoms)
        flat = [a for mol in mols for a in mol]
        assert len(flat) == len(set(flat)) == len(atoms)

    def test_single_water_molecule(self):
        mols = build_molecules(water_atoms())
        assert len(mols) == 1
        assert set(mols[0]) == {0, 1, 2}

    def test_two_water_molecules_separate(self):
        mols = build_molecules(two_water_atoms())
        assert len(mols) == 2

    def test_isolated_atoms_each_own_group(self):
        mols = build_molecules(isolated_atoms())
        assert len(mols) == 4
        for mol in mols:
            assert len(mol) == 1

    def test_molecular_mode_returns_covalent_molecules(self):
        mols = build_molecules(two_water_atoms(), mode="molecular")
        assert len(mols) == 2

    def test_bond_scale_zero_gives_all_isolated(self):
        # With scale=0 no atom has any cutoff, every atom is its own group
        mols = build_molecules(water_atoms(), mode="molecular", bond_scale=0.0)
        assert len(mols) == 3

    def test_bond_scale_large_merges_close_molecules(self):
        # Very large scale: two waters at 8 Å should be merged
        mols = build_molecules(two_water_atoms(), mode="molecular", bond_scale=10.0)
        assert len(mols) == 1

    def test_coordination_mode_runs_without_error(self):
        mols = build_molecules(two_water_atoms(), mode="coordination")
        assert len(mols) >= 1

    def test_indices_are_ints(self):
        mols = build_molecules(water_atoms())
        for mol in mols:
            for idx in mol:
                assert isinstance(idx, int)

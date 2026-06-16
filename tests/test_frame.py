"""Tests for the Frame class (chempiler.frame)."""

import numpy as np
import pytest
from ase import Atoms

from chempiler.frame import Frame


def make_frame(symbols, positions, molecules=None, cell=(100., 100., 100.), pbc=False):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    if molecules is None:
        molecules = [[i] for i in range(len(symbols))]
    frame = Frame(atoms=atoms, molecules=molecules)
    frame.build()
    return frame


# ============================================================
# Frame.build — symbols and positions
# ============================================================

class TestFrameBuildSymbolsPositions:

    def test_symbols_populated(self):
        f = make_frame(["O", "H"], [[0, 0, 0], [1, 0, 0]])
        assert f.symbols == ["O", "H"]

    def test_positions_populated(self):
        pos = [[0, 0, 0], [1, 0, 0]]
        f = make_frame(["O", "H"], pos)
        np.testing.assert_allclose(f.positions, pos)

    def test_positions_shape(self):
        f = make_frame(["O", "H", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert f.positions.shape == (3, 3)


# ============================================================
# Frame.build — formula generation
# ============================================================

class TestFrameFormulas:

    def test_h2o_formula(self):
        f = make_frame(["O", "H", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                       molecules=[[0, 1, 2]])
        assert f.formulas == ["H2O"]

    def test_ho_formula(self):
        f = make_frame(["O", "H"], [[0, 0, 0], [1, 0, 0]], molecules=[[0, 1]])
        assert f.formulas == ["HO"]

    def test_single_atom_formula(self):
        f = make_frame(["O"], [[0, 0, 0]])
        assert f.formulas == ["O"]

    def test_formula_sorted_alphabetically(self):
        # H before O → "HO" not "OH"
        f = make_frame(["O", "H"], [[0, 0, 0], [1, 0, 0]], molecules=[[0, 1]])
        assert f.formulas[0] == "HO"

    def test_element_count_in_formula(self):
        # 4 O + 1 Be → "BeO4"
        pos = [[i, 0, 0] for i in range(5)]
        f = make_frame(["Be", "O", "O", "O", "O"], pos,
                       molecules=[[0, 1, 2, 3, 4]])
        assert "Be" in f.formulas[0] and "O4" in f.formulas[0]

    def test_formulas_length_equals_number_of_molecules(self):
        f = make_frame(
            ["O", "H", "H", "O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        assert len(f.formulas) == 2

    def test_two_molecules_both_h2o(self):
        f = make_frame(
            ["O", "H", "H", "O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        assert f.formulas == ["H2O", "H2O"]


# ============================================================
# Frame.build — formula_to_mols and coms
# ============================================================

class TestFrameFormulaToMols:

    def test_formula_to_mols_keys(self):
        f = make_frame(
            ["O", "H", "H", "O", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0]],
            molecules=[[0, 1, 2], [3, 4]],
        )
        assert "H2O" in f.formula_to_mols
        assert "HO" in f.formula_to_mols

    def test_formula_to_mols_indices_valid(self):
        f = make_frame(
            ["O", "H", "H", "O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        for mol_idx in f.formula_to_mols["H2O"]:
            assert 0 <= mol_idx < len(f.molecules)

    def test_com_is_arithmetic_mean(self):
        f = make_frame(["O", "H"], [[0, 0, 0], [2, 0, 0]], molecules=[[0, 1]])
        np.testing.assert_allclose(f.coms[0], [1.0, 0.0, 0.0])

    def test_com_count_equals_molecule_count(self):
        f = make_frame(
            ["O", "H", "H", "O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        assert len(f.coms) == 2


# ============================================================
# Frame.build — atom_to_mol
# ============================================================

class TestFrameAtomToMol:

    def test_atom_to_mol_shape(self):
        f = make_frame(["O", "H", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                       molecules=[[0, 1, 2]])
        assert len(f.atom_to_mol) == 3

    def test_atom_to_mol_values(self):
        f = make_frame(
            ["O", "H", "H", "O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        assert f.atom_to_mol[0] == 0
        assert f.atom_to_mol[1] == 0
        assert f.atom_to_mol[3] == 1
        assert f.atom_to_mol[5] == 1

    def test_all_atoms_assigned(self):
        f = make_frame(
            ["O", "H", "H", "O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        assert (f.atom_to_mol >= 0).all()


# ============================================================
# Frame.build — idempotency and error handling
# ============================================================

class TestFrameBuildMisc:

    def test_build_idempotent(self):
        f = make_frame(["O", "H", "H"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                       molecules=[[0, 1, 2]])
        f.build()
        assert f.formulas == ["H2O"]

    def test_build_returns_self(self):
        atoms = Atoms(symbols=["O"], positions=[[0, 0, 0]])
        f = Frame(atoms=atoms, molecules=[[0]])
        assert f.build() is f

    def test_raises_without_atoms(self):
        f = Frame()
        f.molecules = [[0]]
        with pytest.raises(ValueError):
            f.build()

    def test_raises_without_molecules(self):
        f = Frame()
        f.atoms = Atoms(symbols=["O"], positions=[[0, 0, 0]])
        with pytest.raises(ValueError):
            f.build()


# ============================================================
# Frame.validate
# ============================================================

class TestFrameValidate:

    def test_valid_frame_returns_true(self):
        f = make_frame(["O"], [[0, 0, 0]])
        assert f.validate() is True

    def test_unbuilt_frame_raises(self):
        f = Frame()
        with pytest.raises(RuntimeError):
            f.validate()


# ============================================================
# Frame.mol_of_atom and atoms_in_mol
# ============================================================

class TestFrameLookups:

    def test_mol_of_atom_correct(self):
        f = make_frame(
            ["O", "H", "H", "O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]],
            molecules=[[0, 1, 2], [3, 4, 5]],
        )
        assert f.mol_of_atom(0) == 0
        assert f.mol_of_atom(3) == 1

    def test_mol_of_atom_out_of_range_returns_minus_one(self):
        f = make_frame(["O"], [[0, 0, 0]])
        assert f.mol_of_atom(99) == -1

    def test_atoms_in_mol_returns_correct_indices(self):
        f = make_frame(
            ["O", "H", "H"],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            molecules=[[0, 1, 2]],
        )
        assert sorted(f.atoms_in_mol(0)) == [0, 1, 2]

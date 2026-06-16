"""Tests for atom/molecule selectors (chempiler.selectors)."""

import numpy as np
import pytest
from ase import Atoms

from chempiler.frame import Frame
from chempiler.selectors import atoms, molecules, formulas, resolve


# ============================================================
# helpers
# ============================================================

def make_frame(syms, positions, mols):
    a = Atoms(symbols=syms, positions=positions, cell=[30.] * 3, pbc=False)
    f = Frame(atoms=a, molecules=mols)
    f.build()
    return f


def mixed_frame():
    """Frame containing one H2O and one HO molecule."""
    return make_frame(
        ["O", "H", "H", "O", "H"],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0]],
        [[0, 1, 2], [3, 4]],
    )


# ============================================================
# atoms()
# ============================================================

class TestAtoms:

    def test_no_symbol_returns_all_indices(self):
        f = mixed_frame()
        idx = atoms(f)
        assert set(idx) == {0, 1, 2, 3, 4}

    def test_symbol_filter_oxygen(self):
        f = mixed_frame()
        idx = atoms(f, "O")
        assert set(idx) == {0, 3}

    def test_symbol_filter_hydrogen(self):
        f = mixed_frame()
        idx = atoms(f, "H")
        assert set(idx) == {1, 2, 4}

    def test_nonexistent_symbol_returns_empty(self):
        f = mixed_frame()
        idx = atoms(f, "Be")
        assert len(idx) == 0

    def test_returns_int32(self):
        f = mixed_frame()
        assert atoms(f).dtype == np.int32
        assert atoms(f, "O").dtype == np.int32


# ============================================================
# molecules()
# ============================================================

class TestMolecules:

    def test_returns_all_molecule_indices(self):
        f = mixed_frame()
        idx = molecules(f)
        assert set(idx) == {0, 1}

    def test_single_molecule(self):
        f = make_frame(["O", "H", "H"],
                       [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        idx = molecules(f)
        assert list(idx) == [0]

    def test_returns_int32(self):
        f = mixed_frame()
        assert molecules(f).dtype == np.int32


# ============================================================
# formulas()
# ============================================================

class TestFormulas:

    def test_none_returns_all_molecules(self):
        f = mixed_frame()
        idx = formulas(f)
        assert set(idx) == {0, 1}

    def test_formula_filter_h2o(self):
        f = mixed_frame()
        idx = formulas(f, ["H2O"])
        assert list(idx) == [0]

    def test_formula_filter_ho(self):
        f = mixed_frame()
        idx = formulas(f, ["HO"])
        assert list(idx) == [1]

    def test_both_formulas(self):
        f = mixed_frame()
        idx = formulas(f, ["H2O", "HO"])
        assert set(idx) == {0, 1}

    def test_nonexistent_formula_returns_empty(self):
        f = mixed_frame()
        idx = formulas(f, ["BeH8O4"])
        assert len(idx) == 0

    def test_returns_int32(self):
        f = mixed_frame()
        assert formulas(f, ["H2O"]).dtype == np.int32


# ============================================================
# resolve()
# ============================================================

class TestResolve:

    def test_string_selector_returns_element_atoms(self):
        f = mixed_frame()
        idx = resolve(f, "O")
        assert set(idx) == {0, 3}

    def test_string_selector_hydrogen(self):
        f = mixed_frame()
        idx = resolve(f, "H")
        assert set(idx) == {1, 2, 4}

    def test_dict_selector_formula_and_element(self):
        f = mixed_frame()
        idx = resolve(f, {"H2O": "O"})
        # Only the O atom in H2O (atom 0)
        assert set(idx) == {0}

    def test_dict_selector_formula_and_element_h(self):
        f = mixed_frame()
        idx = resolve(f, {"H2O": "H"})
        # H atoms 1 and 2 belong to H2O
        assert set(idx) == {1, 2}

    def test_dict_selector_none_element_returns_all_in_formula(self):
        f = mixed_frame()
        idx = resolve(f, {"H2O": None})
        assert set(idx) == {0, 1, 2}

    def test_dict_selector_absent_formula_returns_empty(self):
        f = mixed_frame()
        idx = resolve(f, {"BeH8O4": "O"})
        assert len(idx) == 0

    def test_invalid_selector_raises(self):
        f = mixed_frame()
        with pytest.raises(ValueError):
            resolve(f, 42)

    def test_returns_int32(self):
        f = mixed_frame()
        assert resolve(f, "O").dtype == np.int32
        assert resolve(f, {"H2O": "O"}).dtype == np.int32

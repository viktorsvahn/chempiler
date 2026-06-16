"""Tests for the HDF5 cache module (chempiler.cache)."""

import numpy as np
import pytest
from ase import Atoms

from chempiler.frame import Frame
from chempiler.cache import make_cache_key, save_cache, load_cache


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules=None, cell=(20., 20., 20.)):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    if molecules is None:
        molecules = [[i] for i in range(len(symbols))]
    f = Frame(atoms=atoms, molecules=molecules)
    f.build()
    return f


def h2o_frame():
    return make_frame(
        ["O", "H", "H"],
        [[5.0, 5.0, 5.0], [5.96, 5.0, 5.0], [5.0, 5.96, 5.0]],
        molecules=[[0, 1, 2]],
    )


def ho_frame():
    return make_frame(
        ["O", "H"],
        [[5.0, 5.0, 5.0], [5.96, 5.0, 5.0]],
        molecules=[[0, 1]],
    )


# ============================================================
# make_cache_key
# ============================================================

class TestMakeCacheKey:

    def test_returns_hex_string(self):
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        assert isinstance(key, str)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic(self):
        key1 = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        key2 = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        assert key1 == key2

    def test_different_file_gives_different_key(self):
        k1 = make_cache_key("a.xyz", "molecular", 1.0, 1.3, None)
        k2 = make_cache_key("b.xyz", "molecular", 1.0, 1.3, None)
        assert k1 != k2

    def test_different_mode_gives_different_key(self):
        k1 = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        k2 = make_cache_key("traj.xyz", "sphere", 1.0, 1.3, None)
        assert k1 != k2

    def test_different_scale_gives_different_key(self):
        k1 = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        k2 = make_cache_key("traj.xyz", "molecular", 1.1, 1.3, None)
        assert k1 != k2

    def test_different_max_frames_gives_different_key(self):
        k1 = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, 100)
        k2 = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, 200)
        assert k1 != k2

    def test_sphere_cutoffs_included(self):
        k1 = make_cache_key("traj.xyz", "sphere", 1.0, 1.3, None, None)
        k2 = make_cache_key("traj.xyz", "sphere", 1.0, 1.3, None,
                             {("Be", "O"): 2.4})
        assert k1 != k2

    def test_sphere_cutoffs_order_independent(self):
        k1 = make_cache_key("t.xyz", "sphere", 1.0, 1.3, None,
                             {("Be", "O"): 2.4, ("Be", "Cl"): 2.8})
        k2 = make_cache_key("t.xyz", "sphere", 1.0, 1.3, None,
                             {("Be", "Cl"): 2.8, ("Be", "O"): 2.4})
        assert k1 == k2


# ============================================================
# save_cache / load_cache — round-trip
# ============================================================

class TestCacheRoundTrip:

    def test_single_frame_round_trip(self, tmp_path):
        frames = [h2o_frame()]
        path = str(tmp_path / "test.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, frames, key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        assert len(loaded) == 1

    def test_multiple_frames_preserved(self, tmp_path):
        frames = [h2o_frame(), ho_frame(), h2o_frame()]
        path = str(tmp_path / "multi.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, frames, key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        assert len(loaded) == 3

    def test_formulas_preserved(self, tmp_path):
        frames = [h2o_frame(), ho_frame()]
        path = str(tmp_path / "formulas.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, frames, key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        assert loaded[0].formulas == ["H2O"]
        assert loaded[1].formulas == ["HO"]

    def test_molecules_preserved(self, tmp_path):
        f = h2o_frame()
        path = str(tmp_path / "mols.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, [f], key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        assert sorted(loaded[0].molecules[0]) == sorted(f.molecules[0])

    def test_symbols_restored(self, tmp_path):
        f = h2o_frame()
        path = str(tmp_path / "syms.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, [f], key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        assert loaded[0].symbols == f.symbols

    def test_positions_restored(self, tmp_path):
        f = h2o_frame()
        path = str(tmp_path / "pos.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, [f], key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        np.testing.assert_allclose(loaded[0].positions, f.positions)

    def test_coms_restored(self, tmp_path):
        f = h2o_frame()
        path = str(tmp_path / "coms.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, [f], key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        np.testing.assert_allclose(
            np.array(loaded[0].coms), np.array(f.coms), atol=1e-6
        )

    def test_formula_to_mols_restored(self, tmp_path):
        f = h2o_frame()
        path = str(tmp_path / "f2m.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, [f], key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        assert "H2O" in loaded[0].formula_to_mols

    def test_atom_to_mol_restored(self, tmp_path):
        f = h2o_frame()
        path = str(tmp_path / "a2m.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, [f], key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        assert loaded[0].atom_to_mol is not None
        assert len(loaded[0].atom_to_mol) == len(f.atoms)

    def test_frame_order_preserved(self, tmp_path):
        frames = [h2o_frame(), ho_frame(), h2o_frame()]
        path = str(tmp_path / "order.h5")
        key = make_cache_key("traj.xyz", "molecular", 1.0, 1.3, None)
        save_cache(path, frames, key, "molecular", 1.0, 1.3)
        loaded = load_cache(path)
        for orig, loaded_f in zip(frames, loaded):
            assert orig.formulas == loaded_f.formulas


# ============================================================
# load_cache — error handling
# ============================================================

class TestLoadCacheErrors:

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            load_cache(str(tmp_path / "does_not_exist.h5"))

"""Tests for ChempilerTrajectory methods and the _make_vacuum helper.

Unit tests build synthetic frames and inject them directly into a
ChempilerTrajectory instance (bypassing build()) so that extract_segments,
extract_transition, summary, and _make_vacuum can be exercised without needing
real trajectory files.

Integration tests load the real 39-water trajectory to verify build(), summary(),
and the extract methods end-to-end.
"""

import numpy as np
import pytest
from pathlib import Path
from ase import Atoms
from ase.io import read as ase_read

from chempiler import ChempilerTrajectory
from chempiler.frame import Frame
from chempiler.trajectory import _make_vacuum

TESTS_DIR = Path(__file__).parent


# ============================================================
# helpers
# ============================================================

def make_frame(symbols, positions, molecules=None,
               cell=(20., 20., 20.), pbc=True):
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    if molecules is None:
        molecules = [[i] for i in range(len(symbols))]
    f = Frame(atoms=atoms, molecules=molecules)
    f.build()
    return f


def h2o_frame(ox=5.0, oy=5.0, oz=5.0):
    return make_frame(
        ["O", "H", "H"],
        [[ox, oy, oz], [ox + 0.96, oy, oz], [ox, oy + 0.96, oz]],
        [[0, 1, 2]],
    )


def ho_frame():
    return make_frame(
        ["O", "H"],
        [[5.0, 5.0, 5.0], [5.96, 5.0, 5.0]],
        [[0, 1]],
    )


def mixed_frame():
    """Frame with one H2O and one HO molecule."""
    return make_frame(
        ["O", "H", "H", "O", "H"],
        [[5., 5., 5.], [5.96, 5., 5.], [5., 5.96, 5.],
         [12., 5., 5.], [12.96, 5., 5.]],
        [[0, 1, 2], [3, 4]],
    )


def make_traj(frames):
    """Inject synthetic frames into a ChempilerTrajectory bypassing build()."""
    t = ChempilerTrajectory("dummy.traj")
    t.frames = frames
    return t


# ============================================================
# _make_vacuum — basic output properties
# ============================================================

class TestMakeVacuum:

    def test_pbc_removed(self):
        f = h2o_frame()
        result = _make_vacuum(f)
        assert not result.get_pbc().any()

    def test_no_cell(self):
        f = h2o_frame()
        result = _make_vacuum(f)
        assert (np.asarray(result.get_cell()) == 0).all()

    def test_symbols_preserved(self):
        f = h2o_frame()
        result = _make_vacuum(f)
        assert result.get_chemical_symbols() == f.atoms.get_chemical_symbols()

    def test_atom_count_preserved(self):
        f = h2o_frame()
        assert len(_make_vacuum(f)) == len(f.atoms)

    def test_non_periodic_molecule_unchanged(self):
        # Molecule not split across boundary → positions should be unchanged
        f = h2o_frame()
        result = _make_vacuum(f)
        np.testing.assert_allclose(result.get_positions(),
                                   f.atoms.get_positions(), atol=1e-10)


# ============================================================
# _make_vacuum — PBC reassembly
# ============================================================

class TestMakeVacuumPBC:

    def test_split_molecule_reassembled(self):
        """O at x=0.1, H at x=4.9 in 5 Å box → MIC distance 0.2 Å.
        After _make_vacuum the bond should be ~0.2 Å, not 4.8 Å."""
        cell = 5.0
        atoms = Atoms(
            symbols=["O", "H"],
            positions=[[0.1, 0, 0], [4.9, 0, 0]],
            cell=[cell] * 3,
            pbc=True,
        )
        f = Frame(atoms=atoms, molecules=[[0, 1]])
        f.build()
        result = _make_vacuum(f)
        pos = result.get_positions()
        dist = np.linalg.norm(pos[1] - pos[0])
        assert dist < 1.0   # should be ~0.2 Å, definitely not 4.8 Å

    def test_whole_molecule_not_displaced(self):
        """A molecule already whole should not be moved."""
        f = h2o_frame()
        expected = f.atoms.get_positions().copy()
        result = _make_vacuum(f)
        np.testing.assert_allclose(result.get_positions(), expected, atol=1e-10)

    def test_single_atom_molecule_unaffected(self):
        atoms = Atoms(symbols=["O"], positions=[[2.0, 2.0, 2.0]],
                      cell=[10.] * 3, pbc=True)
        f = Frame(atoms=atoms, molecules=[[0]])
        f.build()
        result = _make_vacuum(f)
        np.testing.assert_allclose(result.get_positions()[0], [2.0, 2.0, 2.0])


# ============================================================
# _make_vacuum — centering
# ============================================================

class TestMakeVacuumCenter:

    def test_centering_puts_target_at_origin(self):
        """HO molecule at x=10 → after centering, its centroid should be ~0."""
        f = ho_frame()
        result = _make_vacuum(f, center_formula="HO")
        pos = result.get_positions()
        mol_idx = f.formula_to_mols["HO"][0]
        atom_indices = f.molecules[mol_idx]
        centroid = pos[atom_indices].mean(axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-10)

    def test_centering_absent_formula_no_crash(self):
        """If the formula is absent, _make_vacuum should run without raising."""
        f = h2o_frame()
        _make_vacuum(f, center_formula="HO")  # HO not in this frame

    def test_no_centering_without_flag(self):
        """Without center_formula, positions are returned as-is (for whole mol)."""
        f = h2o_frame()
        expected = f.atoms.get_positions().copy()
        result = _make_vacuum(f, center_formula=None)
        np.testing.assert_allclose(result.get_positions(), expected, atol=1e-10)


# ============================================================
# ChempilerTrajectory.summary
# ============================================================

class TestSummary:

    def test_formula_counts_correct(self):
        t = make_traj([h2o_frame()] * 3 + [ho_frame()] * 2)
        s = t.summary()
        assert s["H2O"] == 3
        assert s["HO"] == 2

    def test_formula_count_sums_per_frame(self):
        # mixed_frame has 1 H2O and 1 HO → both should be counted
        t = make_traj([mixed_frame()] * 4)
        s = t.summary()
        assert s["H2O"] == 4
        assert s["HO"] == 4

    def test_empty_frames_returns_empty_dict(self):
        t = make_traj([])
        assert t.summary() == {}

    def test_returns_dict(self):
        t = make_traj([h2o_frame()])
        assert isinstance(t.summary(), dict)


# ============================================================
# extract_segments — file naming and return value
# ============================================================

class TestExtractSegments:

    def test_returns_list_of_tuples(self, tmp_path):
        t = make_traj([h2o_frame()] * 5 + [ho_frame()] * 3 + [h2o_frame()] * 5)
        segs = t.extract_segments("HO", output_dir=str(tmp_path))
        assert isinstance(segs, list)
        assert all(isinstance(s, tuple) and len(s) == 2 for s in segs)

    def test_file_created_with_correct_name(self, tmp_path):
        t = make_traj([mixed_frame()] * 5)
        t.extract_segments("HO", output_dir=str(tmp_path))
        assert (tmp_path / "HO_0_5.xyz").exists()

    def test_two_segments_two_files(self, tmp_path):
        frames = [mixed_frame()] * 3 + [h2o_frame()] * 2 + [mixed_frame()] * 3
        t = make_traj(frames)
        segs = t.extract_segments("HO", output_dir=str(tmp_path))
        assert len(segs) == 2
        for start, end in segs:
            assert (tmp_path / f"HO_{start}_{end}.xyz").exists()

    def test_absent_formula_raises(self, tmp_path):
        t = make_traj([h2o_frame()] * 5)
        with pytest.raises(ValueError):
            t.extract_segments("BeH8O4", output_dir=str(tmp_path))

    def test_output_dir_created(self, tmp_path):
        t = make_traj([mixed_frame()] * 3)
        out = tmp_path / "sub" / "dir"
        t.extract_segments("HO", output_dir=str(out))
        assert out.exists()

    def test_xyz_file_is_readable(self, tmp_path):
        t = make_traj([mixed_frame()] * 4)
        segs = t.extract_segments("HO", output_dir=str(tmp_path))
        start, end = segs[0]
        xyz_path = tmp_path / f"HO_{start}_{end}.xyz"
        loaded = ase_read(str(xyz_path), index=":")
        assert len(loaded) == end - start

    def test_vacuum_flag_creates_file(self, tmp_path):
        t = make_traj([mixed_frame()] * 3)
        t.extract_segments("HO", output_dir=str(tmp_path), vacuum=True)
        assert list(tmp_path.glob("HO_*.xyz"))

    def test_vacuum_output_has_no_pbc(self, tmp_path):
        t = make_traj([mixed_frame()] * 3)
        segs = t.extract_segments("HO", output_dir=str(tmp_path), vacuum=True)
        start, end = segs[0]
        loaded = ase_read(str(tmp_path / f"HO_{start}_{end}.xyz"), index="0")
        assert not loaded.get_pbc().any()

    def test_center_flag_creates_file(self, tmp_path):
        t = make_traj([mixed_frame()] * 3)
        t.extract_segments("HO", output_dir=str(tmp_path),
                           vacuum=True, center=True)
        assert list(tmp_path.glob("HO_*.xyz"))


# ============================================================
# extract_transition — file naming, event types, clipping
# ============================================================

class TestExtractTransition:

    def _ho_traj(self):
        """HO appears in frames 5-9, then again in 15-19."""
        return make_traj(
            [h2o_frame()] * 5
            + [mixed_frame()] * 5
            + [h2o_frame()] * 5
            + [mixed_frame()] * 5
        )

    def test_birth_file_created(self, tmp_path):
        t = self._ho_traj()
        t.extract_transition("HO", segment=0, buffer=3, event="birth",
                              output_dir=str(tmp_path))
        assert (tmp_path / "HO_seg0_birth.xyz").exists()

    def test_death_file_created(self, tmp_path):
        t = self._ho_traj()
        t.extract_transition("HO", segment=0, buffer=3, event="death",
                              output_dir=str(tmp_path))
        assert (tmp_path / "HO_seg0_death.xyz").exists()

    def test_both_creates_two_files(self, tmp_path):
        t = self._ho_traj()
        t.extract_transition("HO", segment=0, buffer=3, event="both",
                              output_dir=str(tmp_path))
        assert (tmp_path / "HO_seg0_birth.xyz").exists()
        assert (tmp_path / "HO_seg0_death.xyz").exists()

    def test_birth_window_width(self, tmp_path):
        """Birth window = [start-buffer, start+buffer], so ~2*buffer frames."""
        t = self._ho_traj()
        t.extract_transition("HO", segment=0, buffer=3, event="birth",
                              output_dir=str(tmp_path))
        loaded = ase_read(str(tmp_path / "HO_seg0_birth.xyz"), index=":")
        # HO birth at frame 5; buffer=3 → [2, 8] → 6 frames
        assert len(loaded) == 6

    def test_window_clips_at_start_of_trajectory(self, tmp_path):
        """Segment starting at frame 1 with buffer=5 → window starts at 0."""
        frames = [mixed_frame()] * 3 + [h2o_frame()] * 10
        t = make_traj(frames)
        t.extract_transition("HO", segment=0, buffer=5, event="birth",
                              output_dir=str(tmp_path))
        loaded = ase_read(str(tmp_path / "HO_seg0_birth.xyz"), index=":")
        # start=0, buffer=5 → [max(0,0-5), min(13,0+5)] = [0, 5] → 5 frames
        assert len(loaded) <= 6  # can't exceed trajectory length

    def test_window_clips_at_end_of_trajectory(self, tmp_path):
        """Segment ending at last frame with buffer=5 → window ends at len."""
        frames = [h2o_frame()] * 10 + [mixed_frame()] * 3
        t = make_traj(frames)
        t.extract_transition("HO", segment=0, buffer=5, event="death",
                              output_dir=str(tmp_path))
        loaded = ase_read(str(tmp_path / "HO_seg0_death.xyz"), index=":")
        assert len(loaded) <= 10  # clipped at end

    def test_returns_segment_tuple(self, tmp_path):
        t = self._ho_traj()
        result = t.extract_transition("HO", segment=0, buffer=3,
                                      output_dir=str(tmp_path))
        assert isinstance(result, tuple) and len(result) == 2

    def test_absent_formula_raises(self, tmp_path):
        t = make_traj([h2o_frame()] * 5)
        with pytest.raises(ValueError):
            t.extract_transition("BeH8O4", segment=0,
                                 output_dir=str(tmp_path))

    def test_invalid_segment_index_raises(self, tmp_path):
        t = self._ho_traj()
        with pytest.raises(IndexError):
            t.extract_transition("HO", segment=99, output_dir=str(tmp_path))

    def test_second_segment_index(self, tmp_path):
        t = self._ho_traj()
        t.extract_transition("HO", segment=1, buffer=2, event="birth",
                              output_dir=str(tmp_path))
        assert (tmp_path / "HO_seg1_birth.xyz").exists()

    def test_vacuum_flag_accepted(self, tmp_path):
        t = self._ho_traj()
        t.extract_transition("HO", segment=0, buffer=2, vacuum=True,
                              output_dir=str(tmp_path))
        assert (tmp_path / "HO_seg0_birth.xyz").exists()

    def test_center_flag_accepted(self, tmp_path):
        t = self._ho_traj()
        t.extract_transition("HO", segment=0, buffer=2, vacuum=True,
                             center=True, output_dir=str(tmp_path))
        assert (tmp_path / "HO_seg0_birth.xyz").exists()


# ============================================================
# integration — real 39-water trajectory
# ============================================================

@pytest.fixture(scope="module")
def water_traj():
    t = ChempilerTrajectory(
        str(TESTS_DIR / "39_water_OH.traj"),
        mode="molecular",
        covalent_scale=1.0,
    )
    t.build(cache_file=str(TESTS_DIR / "cache_file.h5"))
    return t


class TestBuildIntegration:

    def test_frames_loaded(self, water_traj):
        assert len(water_traj.frames) > 0

    def test_all_frames_have_atoms(self, water_traj):
        for f in water_traj.frames:
            assert f.atoms is not None

    def test_summary_contains_h2o(self, water_traj):
        s = water_traj.summary()
        assert "H2O" in s

    def test_summary_counts_positive(self, water_traj):
        s = water_traj.summary()
        for v in s.values():
            assert v > 0


class TestExtractSegmentsIntegration:

    def test_extract_h2o_creates_file(self, water_traj, tmp_path):
        water_traj.extract_segments("H2O", output_dir=str(tmp_path))
        xyz_files = list(tmp_path.glob("H2O_*.xyz"))
        assert len(xyz_files) >= 1

    def test_extract_vacuum_no_pbc(self, water_traj, tmp_path):
        water_traj.extract_segments("H2O", output_dir=str(tmp_path),
                                    vacuum=True)
        xyz_files = list(tmp_path.glob("H2O_*.xyz"))
        atoms = ase_read(str(xyz_files[0]), index="0")
        assert not atoms.get_pbc().any()


class TestExtractTransitionIntegration:

    def test_extract_ho_birth(self, water_traj, tmp_path):
        water_traj.extract_transition("HO", segment=0, buffer=5,
                                      event="birth",
                                      output_dir=str(tmp_path))
        assert (tmp_path / "HO_seg0_birth.xyz").exists()

    def test_birth_file_readable(self, water_traj, tmp_path):
        water_traj.extract_transition("HO", segment=0, buffer=5,
                                      event="birth",
                                      output_dir=str(tmp_path))
        loaded = ase_read(str(tmp_path / "HO_seg0_birth.xyz"), index=":")
        assert len(loaded) > 0

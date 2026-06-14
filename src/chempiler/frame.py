"""Per-frame molecular data model.

A Frame pairs an ASE Atoms object with a molecular topology (list of atom-index
groups) and exposes derived lookup tables rebuilt on every call to build().
Atom indexing is always local to the frame; no cross-frame identity is assumed.
"""

from collections import Counter
import numpy as np


class Frame:
    """A single trajectory frame with molecular topology and derived metadata.

    Parameters
    ----------
    atoms : ase.Atoms, optional
        ASE atomic configuration for this frame.
    molecules : list of list of int, optional
        Each inner list contains the atom indices belonging to one molecule.

    Notes
    -----
    Call build() after setting atoms and molecules to populate all derived
    attributes (symbols, positions, formulas, coms, atom_to_mol).
    """

    def __init__(self, atoms=None, molecules=None):
        self.atoms = atoms
        self.molecules = molecules

        self.symbols = None
        self.positions = None

        self.formulas = []
        self.coms = []
        self.formula_to_mols = {}

        self.atom_to_mol = None

    def build(self):
        """Build all derived metadata from atoms and molecules.

        Populates symbols, positions, formulas, coms, formula_to_mols, and
        atom_to_mol. Safe to call multiple times; each call fully rebuilds
        the derived state.

        Returns
        -------
        Frame
            Returns self for method chaining.

        Raises
        ------
        ValueError
            If atoms or molecules have not been set.
        """
        if self.atoms is None or self.molecules is None:
            raise ValueError("Frame requires atoms and molecules before build().")

        self.symbols = self.atoms.get_chemical_symbols()
        self.positions = self.atoms.get_positions()

        self.formulas = []
        self.coms = []
        self.formula_to_mols = {}

        for mi, mol in enumerate(self.molecules):
            counts = Counter(self.symbols[i] for i in mol)
            formula = "".join(
                f"{k}{v if v > 1 else ''}"
                for k, v in sorted(counts.items())
            )
            self.formulas.append(formula)
            self.formula_to_mols.setdefault(formula, []).append(mi)
            self.coms.append(np.mean(self.positions[mol], axis=0))

        self._build_atom_to_mol()
        return self

    def _build_atom_to_mol(self):
        """Populate atom_to_mol mapping from the current molecules list."""
        n = len(self.atoms)
        self.atom_to_mol = np.full(n, -1, dtype=np.int32)
        for mi, mol in enumerate(self.molecules):
            for a in mol:
                if 0 <= a < n:
                    self.atom_to_mol[a] = mi

    def validate(self):
        """Check internal consistency of the frame.

        Returns
        -------
        bool
            True if the frame is consistent.

        Raises
        ------
        RuntimeError
            If build() has not been called or atom_to_mol is the wrong size.
        """
        if self.atom_to_mol is None:
            raise RuntimeError("Frame not built; call build() first.")
        if len(self.atom_to_mol) != len(self.atoms):
            raise RuntimeError("atom_to_mol size mismatch.")
        return True

    def mol_of_atom(self, atom_index):
        """Return the molecule index containing the given atom, or -1.

        Parameters
        ----------
        atom_index : int

        Returns
        -------
        int
            Molecule index, or -1 if the atom is not assigned to any molecule.
        """
        if atom_index >= len(self.atom_to_mol):
            return -1
        return int(self.atom_to_mol[atom_index])

    def atoms_in_mol(self, mol_id):
        """Return the list of atom indices belonging to molecule mol_id.

        Parameters
        ----------
        mol_id : int

        Returns
        -------
        list of int
        """
        return self.molecules[mol_id]

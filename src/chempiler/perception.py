"""Molecular perception: build covalent/coordination topology from an ASE frame.

Uses ASE natural cutoffs scaled by a user-supplied factor and a Union-Find
algorithm to group atoms into connected components (molecules).
"""

import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs


class UnionFind:
    """Disjoint-set data structure with path compression.

    Parameters
    ----------
    n : int
        Number of elements (labelled 0 .. n-1).
    """

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        """Return the root of the set containing x (with path compression).

        Parameters
        ----------
        x : int

        Returns
        -------
        int
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        """Merge the sets containing a and b.

        Parameters
        ----------
        a, b : int
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_molecules(atoms, mode="molecular", bond_scale=1.0, coordination_scale=1.3):
    """Partition atoms into bonded groups using scaled ASE natural cutoffs.

    Parameters
    ----------
    atoms : ase.Atoms
        Single-frame atomic configuration.
    mode : {"molecular", "coordination"}
        ``"molecular"`` uses covalent cutoffs scaled by *bond_scale*.
        ``"coordination"`` uses larger cutoffs scaled by *coordination_scale*
        to capture first-shell coordination environments.
    bond_scale : float
        Multiplicative scale applied to natural cutoffs in molecular mode.
    coordination_scale : float
        Multiplicative scale applied to natural cutoffs in coordination mode.

    Returns
    -------
    list of list of int
        Each inner list is the set of atom indices belonging to one molecule
        (connected component).
    """
    cutoffs = natural_cutoffs(atoms)

    if mode == "molecular":
        cutoffs = [c * bond_scale for c in cutoffs]
    elif mode == "coordination":
        cutoffs = [c * coordination_scale for c in cutoffs]

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    uf = UnionFind(len(atoms))
    for i in range(len(atoms)):
        neighbours, _ = nl.get_neighbors(i)
        for j in neighbours:
            uf.union(i, j)

    # Group atoms by their UnionFind root to form molecules.
    groups = {}
    for i in range(len(atoms)):
        root = uf.find(i)
        groups.setdefault(root, []).append(int(i))

    return list(groups.values())

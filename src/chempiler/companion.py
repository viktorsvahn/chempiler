"""HDF5 companion file for storing calculator results alongside ASE trajectories.

ASE's ``.traj`` format cannot reliably round-trip non-standard per-atom arrays
(charges, spins, density coefficients, …) — they are silently dropped.  This
module provides a companion HDF5 file that stores *all* ``atoms.calc.results``
entries regardless of shape, synced frame-for-frame with the trajectory.

Cluster side (md.py)::

    from chempiler import CalcResultsWriter

    writer = CalcResultsWriter(output_name + '_polar.h5')

    def write_frame():
        traj.write(atoms)
        writer.write(atoms)

    dyn.attach(write_frame, interval=steps_per_block)

Analysis side::

    traj = ChempilerTrajectory('run.traj')
    traj.build(companion='run_polar.h5')
    # frame.atoms.arrays['charges'], ['spins'], etc. now available

    hists, meta = spatial_density_maps(traj.frames, ..., weights='spins')
"""

import numpy as np


class CalcResultsWriter:
    """Append ``atoms.calc.results`` to an HDF5 companion file per call.

    Every key in ``calc.results`` is stored as a resizable dataset indexed by
    frame number.  The file is opened and closed on each call so a crash
    mid-simulation does not corrupt previously written frames.  Restart
    (append mode) is handled automatically.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.  Created on the first :meth:`write` call.
    """

    def __init__(self, path):
        self.path = path

    def write(self, atoms):
        """Append the current ``atoms.calc.results`` as the next frame."""
        if atoms.calc is None:
            return
        import h5py
        with h5py.File(self.path, 'a') as hf:
            n = int(hf.attrs.get('n_frames', 0))
            for key, val in atoms.calc.results.items():
                arr = np.asarray(val)
                if key not in hf:
                    hf.create_dataset(
                        key,
                        data=arr[np.newaxis],
                        maxshape=(None,) + arr.shape,
                        chunks=True,
                    )
                else:
                    hf[key].resize(n + 1, axis=0)
                    hf[key][n] = arr
            hf.attrs['n_frames'] = n + 1

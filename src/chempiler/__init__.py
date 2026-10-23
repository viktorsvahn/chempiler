"""Chempiler: chemical perception and analysis for reactive MD trajectories."""

from .companion import CalcResultsWriter
from .trajectory import ChempilerTrajectory
from .rdf import plot_rdf, rdf_shells, ShellInfo, ShellEnvironments, load_rdf
from .msd import diffusive_window
from .vanhove import van_hove_heatmap
from .vdos import read_vdos, smooth, plot_vdos
from .segmentation import stitch_identity_tracks
from .view import show, view3d, plot_structure
from .sdf import (spatial_density_map, spatial_density_maps,
                   spatial_density_3d, sdf_radial_profile, sdf_region_membership,
                   plot_sdf, plot_sdf_maps, plot_sdf_panel,
                   plot_sdf_3d,
                   molecule_overlay, gto_kernel, SCHOENFLIES)
from .colours import ELEM_COLOR, elem_color, elem_radius, draw_atom_2d, draw_molecule_2d
from .core.statistics import (estimate_autocorrelation_time, confidence_halfwidth,
                              blocked_stderr)

__all__ = ["CalcResultsWriter", "ChempilerTrajectory", "plot_rdf", "rdf_shells", "ShellInfo", "ShellEnvironments", "load_rdf",
           "diffusive_window", "van_hove_heatmap", "read_vdos", "smooth", "plot_vdos",
           "stitch_identity_tracks", "show", "view3d", "plot_structure",
           "spatial_density_map", "spatial_density_maps", "spatial_density_3d", "sdf_radial_profile", "sdf_region_membership",
           "plot_sdf", "plot_sdf_maps", "plot_sdf_panel", "plot_sdf_3d",
           "molecule_overlay", "gto_kernel", "SCHOENFLIES",
           "ELEM_COLOR", "elem_color", "elem_radius", "draw_atom_2d", "draw_molecule_2d",
           "estimate_autocorrelation_time", "confidence_halfwidth", "blocked_stderr"]

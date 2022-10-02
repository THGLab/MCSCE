import numpy as np
from mcsce.libs.libcalc import calc_angle, calc_torsion_angles, rotate_coordinates_Q
from mcsce.core.build_definitions import sidechain_templates
from mcsce.libs.libscbuild import rotate_sidechain

template_coord, _ = rotate_sidechain("PTR", np.array([123, 135])) #, -67, 118, -5]))
tempdb = sidechain_templates["PTR"][0]
tempdb.coords = template_coord
tempdb.write_PDB("rotated_PTR.pdb")

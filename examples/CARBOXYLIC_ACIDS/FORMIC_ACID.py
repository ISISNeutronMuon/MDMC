from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined

C1  = Atom('C', position = [-4.711,   0.956,   0.000], name = '853',   charge =  0.632, atom_type = 1) 
H1  = Atom('C', position = [-5.832,   1.099,   0.000], name = ' 91',  charge =  0.086, atom_type = 2)
O1  = Atom('C', position = [-4.032,   1.962,   0.000], name = '854',  charge = -0.546, atom_type = 3)
O2  = Atom('C', position = [-4.074,  -0.186,  -0.000], name = ' 96',  charge = -0.615, atom_type = 4)
HO  = Atom('C', position = [-3.106,  -0.003,  -0.000], name = ' 97',  charge =  0.443, atom_type = 5)

# BONDS
C1H1_bond = Bond(C1, H1)
C1O1_bond = Bond(C1, O1)
C1O2_bond = Bond(C1, O2)
O2HO_bond = Bond(O2, HO)

# ANGLES
C1O2HO_angle = BondAngle((C1, O2, HO))
H1C1O1_angle = BondAngle((H1, C1, O1))
H1C1O2_angle = BondAngle((H1, C1, O2))
O1C1O2_angle = BondAngle((O1, C1, O2))

# DIHEDRALS - PROPER
H1C1O2HO_dihedral = DihedralAngle((H1, C1, O2, HO))
O1C1O2HO_dihedral = DihedralAngle((O1, C1, O2, HO))

# DIHEDRALS - IMPROPER
O2C1O1H1_dihedral = DihedralAngle(atoms=[O2, C1, O1, H1], improper=True)

# DISPERSION
C1C1_disp = Dispersion(universe, (1, 1), cutoff = 8.0, vdw_tail_correction=True)
H1H1_disp = Dispersion(universe, (2, 2), cutoff = 8.0, vdw_tail_correction=True)
O1O1_disp = Dispersion(universe, (3, 3), cutoff = 8.0, vdw_tail_correction=True)
O2O2_disp = Dispersion(universe, (4, 4), cutoff = 8.0, vdw_tail_correction=True)
HOHO_disp = Dispersion(universe, (5, 5), cutoff = 8.0, vdw_tail_correction=True)



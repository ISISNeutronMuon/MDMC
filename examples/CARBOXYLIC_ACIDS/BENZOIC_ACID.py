from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined

C1  = Atom('C', position = [-5.722,   2.289,  -0.000], name = '205',   charge = -0.083, atom_type = 1) 
C2  = Atom('C', position = [-4.493,   2.959,  -0.000], name = '205',  charge = -0.139, atom_type = 2)
C3 = Atom('C', position = [-3.298,   2.242,   0.000],  name = '205',  charge = -0.096, atom_type = 3)
C4 = Atom('C', position = [-3.325,   0.851,  -0.000],  name = '205',  charge = -0.139, atom_type = 4)
C5 = Atom('C', position = [-4.549,   0.179,   0.000],  name = '205',  charge = -0.083, atom_type = 5)
C6  = Atom('C', position = [-5.763,   0.887,  -0.000], name = '205',  charge = -0.173, atom_type = 6)
H1  = Atom('H', position = [-6.614,   2.904,  -0.000], name = ' 91',  charge =  0.146, atom_type = 7) 
H2  = Atom('H', position = [-4.466,   4.047,  -0.000], name = ' 91',    charge =  0.142, atom_type = 8)
H3  = Atom('H', position = [-2.346,   2.767,   0.000], name = ' 91',  charge =  0.140, atom_type = 9)
H4 = Atom('H', position = [-2.396,   0.286,   0.000],  name = ' 91',  charge =  0.142, atom_type = 10)
H5 = Atom('H', position = [-4.552,  -0.911,  -0.000],  name = ' 91',  charge =  0.146, atom_type = 11)
C7 = Atom('C', position = [-7.017,   0.073,  -0.000],  name = '853',  charge =  0.642, atom_type = 12)
O1  = Atom('O', position = [-6.997,  -1.148,  -0.000], name = '854',  charge = -0.493, atom_type = 13)
O2  = Atom('O', position = [-8.180,   0.755,   0.000], name = ' 96',  charge = -0.576, atom_type = 14) 
HO  = Atom('H', position = [-8.134,   1.722,   0.000], name = ' 97',    charge =  0.424, atom_type = 15)

# BONDS
C1C2_bond = Bond(C1,C2)
C1C6_bond = Bond(C1,C6)
C1H1_bond = Bond(C1,H1)
C2C3_bond = Bond(C2,C3)
C2H2_bond = Bond(C2,H2)
C3C4_bond = Bond(C3,C4)
C3H3_bond = Bond(C3,H3)
C4C5_bond = Bond(C4,C5)
C4H4_bond = Bond(C4,H4)
C5C6_bond = Bond(C5,C6)
C5H5_bond = Bond(C5,H5)
C6C7_bond = Bond(C6,C7)
C7O1_bond = Bond(C7,O1)
C7O2_bond = Bond(C7,O2)
O2HO_bond = Bond(O2,HO)

# ANGLES
C1C2C3_angle = BondAngle((C1,C2,C3))
C1C2H2_angle = BondAngle((C1,C2,H2))
C1C6C5_angle = BondAngle((C1,C6,C5))
C1C6C7_angle = BondAngle((C1,C6,C7))
C2C1C6_angle = BondAngle((C2,C1,C6))
C2C1H1_angle = BondAngle((C2,C1,H1))
C2C3C4_angle = BondAngle((C2,C3,C4))
C2C3H3_angle = BondAngle((C2,C3,H3))
C3C2H2_angle = BondAngle((C3,C2,H2))
C3C4C5_angle = BondAngle((C3,C4,C5))
C3C4H4_angle = BondAngle((C3,C4,H4))
C4C3H3_angle = BondAngle((C4,C3,H3))
C4C5C6_angle = BondAngle((C4,C5,C6))
C4C5H5_angle = BondAngle((C4,C5,H5))
C5C4H4_angle = BondAngle((C5,C4,H4))
C5C6C7_angle = BondAngle((C5,C6,C7))
C6C1H1_angle = BondAngle((C6,C1,H1))
C6C5H5_angle = BondAngle((C6,C5,H5))
C6C7O1_angle = BondAngle((C6,C7,O1))
C6C7O2_angle = BondAngle((C6,C7,O2))
C7O2HO_angle = BondAngle((C7,O2,HO))
O1C7O2_angle = BondAngle((O1,C7,O2))

# DIHEDRALS - proper
C1C2C3C4_dihedral = DihedralAngle((C1,C2,C3,C4))
C1C2C3H3_dihedral = DihedralAngle((C1,C2,C3,H3))
C1C6C5C4_dihedral = DihedralAngle((C1,C6,C5,C4))
C1C6C5H5_dihedral = DihedralAngle((C1,C6,C5,H5))
C1C6C7O1_dihedral = DihedralAngle((C1,C6,C7,O1))
C1C6C7O2_dihedral = DihedralAngle((C1,C6,C7,O2))
C2C1C6C5_dihedral = DihedralAngle((C2,C1,C6,C5))
C2C1C6C7_dihedral = DihedralAngle((C2,C1,C6,C7))
C2C3C4C5_dihedral = DihedralAngle((C2,C3,C4,C5))
C2C3C4H4_dihedral = DihedralAngle((C2,C3,C4,H4))
C3C4C5C6_dihedral = DihedralAngle((C3,C4,C5,C6))
C3C4C5H5_dihedral = DihedralAngle((C3,C4,C5,H5))
C4C3C2H2_dihedral = DihedralAngle((C4,C3,C2,H2))
C4C5C6C7_dihedral = DihedralAngle((C4,C5,C6,C7))
C5C4C3H3_dihedral = DihedralAngle((C5,C4,C3,H3))
C5C6C7O1_dihedral = DihedralAngle((C5,C6,C7,O1))
C5C6C7O2_dihedral = DihedralAngle((C5,C6,C7,O2))
C6C1C2C3_dihedral = DihedralAngle((C6,C1,C2,C3))
C6C1C2H2_dihedral = DihedralAngle((C6,C1,C2,H2))
C6C5C4H4_dihedral = DihedralAngle((C6,C5,C4,H4))
C6C7O2HO_dihedral = DihedralAngle((C6,C7,O2,HO))
H1C1C2C3_dihedral = DihedralAngle((H1,C1,C2,C3))
H1C1C2H2_dihedral = DihedralAngle((H1,C1,C2,H2))
H1C1C6C5_dihedral = DihedralAngle((H1,C1,C6,C5))
H1C1C6C7_dihedral = DihedralAngle((H1,C1,C6,C7))
H2C2C3H3_dihedral = DihedralAngle((H2,C2,C3,H3))
H3C3C4H4_dihedral = DihedralAngle((H3,C3,C4,H4))
H4C4C5H5_dihedral = DihedralAngle((H4,C4,C5,H5))
H5C5C6C7_dihedral = DihedralAngle((H5,C5,C6,C7))
O1C7O2HO_dihedral = DihedralAngle((O1,C7,O2,HO))

# DIHEDRALS- improper
C1C3C2H2_dihedral = DihedralAngle(atoms=[C1,C3,C2,H2], improper=True)
C2C4C3H3_dihedral = DihedralAngle(atoms=[C2,C4,C3,H3], improper=True)
C3C5C4H4_dihedral = DihedralAngle(atoms=[C3,C5,C4,H4], improper=True)
C4C6C5H5_dihedral = DihedralAngle(atoms=[C4,C6,C5,H5], improper=True)
C6O1C7O2_dihedral = DihedralAngle(atoms=[C6,O1,C7,O2], improper=True)
H1C1C6C2_dihedral = DihedralAngle(atoms=[H1,C1,C6,C2], improper=True)
C7C1C6C5_dihedral = DihedralAngle(atoms=[C7,C1,C6,C5], improper=True)

# DISPERSION
C1C1_disp = Dispersion(universe, (1,1), cutoff = 8.0, vdw_tail_correction=True)
C2C2_disp = Dispersion(universe, (2,2), cutoff = 8.0, vdw_tail_correction=True)
C3C3_disp = Dispersion(universe, (3,3), cutoff = 8.0, vdw_tail_correction=True)
C4C4_disp = Dispersion(universe, (4,4), cutoff = 8.0, vdw_tail_correction=True)
C5C5_disp = Dispersion(universe, (5,5), cutoff = 8.0, vdw_tail_correction=True)
C6C6_disp = Dispersion(universe, (6,6), cutoff = 8.0, vdw_tail_correction=True)
H1H1_disp = Dispersion(universe, (7,7), cutoff = 8.0, vdw_tail_correction=True)
H2H2_disp = Dispersion(universe, (8,8), cutoff = 8.0, vdw_tail_correction=True)
H3H3_disp = Dispersion(universe, (9,9), cutoff = 8.0, vdw_tail_correction=True)
H4H4_disp = Dispersion(universe, (10,10), cutoff = 8.0, vdw_tail_correction=True)
H5H5_disp = Dispersion(universe, (11,11), cutoff = 8.0, vdw_tail_correction=True)
C7C7_disp = Dispersion(universe, (12,12), cutoff = 8.0, vdw_tail_correction=True)
O1O1_disp = Dispersion(universe, (13,13), cutoff = 8.0, vdw_tail_correction=True)
O2O2_disp = Dispersion(universe, (14,14), cutoff = 8.0, vdw_tail_correction=True)
HOHO_disp = Dispersion(universe, (15,15), cutoff = 8.0, vdw_tail_correction=True)














































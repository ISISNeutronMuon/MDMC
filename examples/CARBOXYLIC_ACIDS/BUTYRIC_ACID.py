from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined

C1  = Atom('C', position = [-6.763,   2.563,  -0.056],  name = ' 80',   charge = -0.094, atom_type = 1) 
C2  = Atom('C', position = [-5.470,   1.724,  -0.068],  name = ' 80',  charge = -0.080, atom_type = 2)
C3  = Atom('C',  position = [-4.213,   2.609,   0.115], name = ' 80',   charge = -0.127, atom_type = 3)
C4  = Atom('C',  position = [-2.888,   1.864,   0.017], name = '853',   charge =  0.635, atom_type = 4)
H1X = Atom('H',  position = [-6.660,   3.449,  -0.687], name = ' 85',   charge =  0.037, atom_type = 5)
H1Y = Atom('H', position = [-7.004,   2.886,   0.959],  name = ' 85',  charge =  0.037, atom_type = 6)
H1Z = Atom('H', position = [-7.599,   1.971,  -0.435],  name = ' 85',  charge =  0.037, atom_type = 7) 
H2X = Atom('H', position = [-5.524,   0.979,   0.730],  name = ' 85',    charge =  0.058, atom_type = 8)
H2Y = Atom('H', position = [-5.401,   1.188,  -1.018],  name = ' 85',  charge =  0.058, atom_type = 9)
H3X = Atom('H',  position = [-4.215,   3.384,  -0.656], name = ' 85',   charge =  0.080, atom_type = 10)
H3Y = Atom('H',  position = [-4.261,   3.113,   1.083], name = ' 85',   charge =  0.080, atom_type = 11)
O1  = Atom('O',  position = [-1.826,   2.429,  -0.133], name = '854',   charge = -0.551, atom_type = 12)
O2  = Atom('O', position = [-2.959,   0.563,   0.120],  name = ' 96',  charge = -0.612, atom_type = 13)
HO  = Atom('H', position = [-2.048,   0.233,   0.029],  name = ' 97',  charge =  0.443, atom_type = 14) 

# BONDS
C1C2_bond = Bond(C1,C2)
C1H1X_bond = Bond(C1,H1X)
C1H1Y_bond = Bond(C1,H1Y)
C1H1Z_bond = Bond(C1,H1Z)
C2C3_bond = Bond(C2,C3)
C2H2X_bond = Bond(C2,H2X)
C2H2Y_bond = Bond(C2,H2Y)
C3C4_bond = Bond(C3,C4)
C3H3X_bond = Bond(C3,H3X)
C3H3Y_bond = Bond(C3,H3Y)
C4O1_bond = Bond(C4,O1)
C4O2_bond = Bond(C4,O2)
O2HO_bond = Bond(O2,HO)

# ANGLES
C1C2C3_angle = BondAngle((C1,C2,C3))
C1C2H2X_angle = BondAngle((C1,C2,H2X))
C1C2H2Y_angle = BondAngle((C1,C2,H2Y))
C2C1H1X_angle = BondAngle((C2,C1,H1X))
C2C1H1Y_angle = BondAngle((C2,C1,H1Y))
C2C1H1Z_angle = BondAngle((C2,C1,H1Z))
C2C3C4_angle = BondAngle((C2,C3,C4))
C2C3H3X_angle = BondAngle((C2,C3,H3X))
C2C3H3Y_angle = BondAngle((C2,C3,H3Y))
C3C2H2X_angle = BondAngle((C3,C2,H2X))
C3C2H2Y_angle = BondAngle((C3,C2,H2Y))
C3C4O1_angle = BondAngle((C3,C4,O1))
C3C4O2_angle = BondAngle((C3,C4,O2))
C4C3H3X_angle = BondAngle((C4,C3,H3X))
C4C3H3Y_angle = BondAngle((C4,C3,H3Y))
C4O2HO_angle = BondAngle((C4,O2,HO))
H1XC1H1Y_angle = BondAngle((H1X,C1,H1Y))
H1XC1H1Z_angle = BondAngle((H1X,C1,H1Z))
H1YC1H1Z_angle = BondAngle((H1Y,C1,H1Z))
H2XC2H2Y_angle = BondAngle((H2X,C2,H2Y))
H3XC3H3Y_angle = BondAngle((H3X,C3,H3Y))
O1C4O2_angle = BondAngle((O1,C4,O2))

# DIHEDRALS - PROPER
C1C2C3C4_dihedral = DihedralAngle((C1,C2,C3,C4))
C1C2C3H3X_dihedral = DihedralAngle((C1,C2,C3,H3X))
C1C2C3H3Y_dihedral = DihedralAngle((C1,C2,C3,H3Y))
C2C3C4O1_dihedral = DihedralAngle((C2,C3,C4,O1))
C2C3C4O2_dihedral = DihedralAngle((C2,C3,C4,O2))
C3C4O2HO_dihedral = DihedralAngle((C3,C4,O2,HO))
C4C3C2H2X_dihedral = DihedralAngle((C4,C3,C2,H2X))
C4C3C2H2Y_dihedral = DihedralAngle((C4,C3,C2,H2Y))
H1XC1C2C3_dihedral = DihedralAngle((H1X,C1,C2,C3))
H1XC1C2H2X_dihedral = DihedralAngle((H1X,C1,C2,H2X))
H1XC1C2H2Y_dihedral = DihedralAngle((H1X,C1,C2,H2Y))
H1YC1C2C3_dihedral = DihedralAngle((H1Y,C1,C2,C3))
H1YC1C2H2X_dihedral = DihedralAngle((H1Y,C1,C2,H2X))
H1YC1C2H2Y_dihedral = DihedralAngle((H1Y,C1,C2,H2Y))
H1ZC1C2C3_dihedral = DihedralAngle((H1Z,C1,C2,C3))
H1ZC1C2H2X_dihedral = DihedralAngle((H1Z,C1,C2,H2X))
H1ZC1C2H2Y_dihedral = DihedralAngle((H1Z,C1,C2,H2Y))
H2XC2C3H3X_dihedral = DihedralAngle((H2X,C2,C3,H3X))
H2XC2C3H3Y_dihedral = DihedralAngle((H2X,C2,C3,H3Y))
H2YC2C3H3X_dihedral = DihedralAngle((H2Y,C2,C3,H3X))
H2YC2C3H3Y_dihedral = DihedralAngle((H2Y,C2,C3,H3Y))
H3XC3C4O1_dihedral = DihedralAngle((H3X,C3,C4,O1))
H3XC3C4O2_dihedral = DihedralAngle((H3X,C3,C4,O2))
H3YC3C4O1_dihedral = DihedralAngle((H3Y,C3,C4,O1))
H3YC3C4O2_dihedral = DihedralAngle((H3Y,C3,C4,O2))
O1C4O2HO_dihedral = DihedralAngle((O1,C4,O2,HO))

# DIHEDRALS - IMPROPER
C3O1C4O2_dihedral = DihedralAngle(atoms=[C3,O1,C4,O2], improper=True)

# DISPERSION
C1C1_disp = Dispersion(universe, (1,1), cutoff = 8.0, vdw_tail_correction=True)
C2C2_disp = Dispersion(universe, (2,2), cutoff = 8.0, vdw_tail_correction=True)
C3C3_disp = Dispersion(universe, (3,3), cutoff = 8.0, vdw_tail_correction=True)
C4C4_disp = Dispersion(universe, (4,4), cutoff = 8.0, vdw_tail_correction=True)
H1XH1X_disp = Dispersion(universe, (5,5), cutoff = 8.0, vdw_tail_correction=True)
H1YH1Y_disp = Dispersion(universe, (6,6), cutoff = 8.0, vdw_tail_correction=True)
H1ZH1Z_disp = Dispersion(universe, (7,7), cutoff = 8.0, vdw_tail_correction=True)
H2XH2X_disp = Dispersion(universe, (8,8), cutoff = 8.0, vdw_tail_correction=True)
H2YH2Y_disp = Dispersion(universe, (9,9), cutoff = 8.0, vdw_tail_correction=True)
H3XH3X_disp = Dispersion(universe, (10,10), cutoff = 8.0, vdw_tail_correction=True)
H3YH3Y_disp = Dispersion(universe, (11,11), cutoff = 8.0, vdw_tail_correction=True)
O1O1_disp = Dispersion(universe, (12,12), cutoff = 8.0, vdw_tail_correction=True)
O2O2_disp = Dispersion(universe, (13,13), cutoff = 8.0, vdw_tail_correction=True)
HOHO_disp = Dispersion(universe, (14,14), cutoff = 8.0, vdw_tail_correction=True)














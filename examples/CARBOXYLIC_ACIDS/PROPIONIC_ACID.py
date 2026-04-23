from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined

C1   = Atom('C', position = [-7.392,   2.907,   0.095], name = ' 80',   charge = -0.095, atom_type = 1) 
C2   = Atom('C', position = [-6.271,   1.848,   0.050], name = ' 80',  charge = -0.126, atom_type = 2)
C3   = Atom('C', position = [-4.891,   2.445,  -0.169], name = '853',  charge =  0.635, atom_type = 3)
H1X  = Atom('H', position = [-7.118,   3.742,   0.743], name = ' 85',  charge =  0.049, atom_type = 4)
H1Y  = Atom('H', position = [-7.594,   3.293,  -0.907], name = ' 85',  charge =  0.049, atom_type = 5)
H1Z  = Atom('H', position = [-8.310,   2.463,   0.484], name = ' 85',   charge =  0.049, atom_type = 6) 
H2X  = Atom('H', position = [-6.483,   1.131,  -0.747], name = ' 85',  charge =  0.078, atom_type = 7)
H2Y  = Atom('H', position = [-6.266,   1.290,   0.989], name = ' 85',  charge =  0.078, atom_type = 8)
O1   = Atom('O', position = [-4.646,   3.631,  -0.221], name = '854',  charge = -0.550, atom_type = 9)
O2   = Atom('O', position = [-3.945,   1.547,  -0.129], name = ' 96',  charge = -0.610, atom_type = 10)
HO   = Atom('H', position = [-3.104,   2.036,  -0.189], name = ' 97',  charge =  0.443, atom_type = 11)

# BONDS
C1C2_bond = Bond(C1, C2)
C1H1X_bond = Bond(C1, H1X)
C1H1Y_bond = Bond(C1, H1Y)
C1H1Z_bond = Bond(C1, H1Z)
C2C3_bond = Bond(C2, C3)
C2H2X_bond = Bond(C2, H2X)
C2H2Y_bond = Bond(C2, H2Y)
C3O1_bond = Bond(C3, O1)
C3O2_bond = Bond(C3, O2)
O2HO_bond = Bond(O2, HO)

# ANGLES
C1C2C3_angle = BondAngle((C1, C2, C3))
C1C2H2X_angle = BondAngle((C1, C2, H2X))
C1C2H2Y_angle = BondAngle((C1, C2, H2Y))
C2C1H1X_angle = BondAngle((C2, C1, H1X))
C2C1H1Y_angle = BondAngle((C2, C1, H1Y))
C2C1H1Z_angle = BondAngle((C2, C1, H1Z))
C2C3O1_angle = BondAngle((C2, C3, O1))
C2C3O2_angle = BondAngle((C2, C3, O2))
C3C2H2X_angle = BondAngle((C3, C2, H2X))
C3C2H2Y_angle = BondAngle((C3, C2, H2Y))
C3O2HO_angle = BondAngle((C3, O2, HO))
H1XC1H1Y_angle = BondAngle((H1X, C1, H1Y))
H1XC1H1Z_angle = BondAngle((H1X, C1, H1Z))
H1YC1H1Z_angle = BondAngle((H1Y, C1, H1Z))
H2XC2H2Y_angle = BondAngle((H2X, C2, H2Y))
O1C3O2_angle = BondAngle((O1, C3, O2))

# DIHEDRALS - PROPER
C1C2C3O1_dihedral = DihedralAngle((C1, C2, C3, O1))
C1C2C3O2_dihedral = DihedralAngle((C1, C2, C3, O2))
C2C3O2HO_dihedral = DihedralAngle((C2, C3, O2, HO))
H1XC1C2C3_dihedral = DihedralAngle((H1X, C1, C2, C3))
H1XC1C2H2X_dihedral = DihedralAngle((H1X, C1, C2, H2X))
H1XC1C2H2Y_dihedral = DihedralAngle((H1X, C1, C2, H2Y))
H1YC1C2C3_dihedral = DihedralAngle((H1Y, C1, C2, C3))
H1YC1C2H2X_dihedral = DihedralAngle((H1Y, C1, C2, H2X))
H1YC1C2H2Y_dihedral = DihedralAngle((H1Y, C1, C2, H2Y))
H1ZC1C2C3_dihedral = DihedralAngle((H1Z, C1, C2, C3))
H1ZC1C2H2X_dihedral = DihedralAngle((H1Z, C1, C2, H2X))
H1ZC1C2H2Y_dihedral = DihedralAngle((H1Z, C1, C2, H2Y))
H2XC2C3O1_dihedral = DihedralAngle((H2X, C2, C3, O1))
H2XC2C3O2_dihedral = DihedralAngle((H2X, C2, C3, O2))
H2YC2C3O1_dihedral = DihedralAngle((H2Y, C2, C3, O1))
H2YC2C3O2_dihedral = DihedralAngle((H2Y, C2, C3, O2))
O1C3O2HO_dihedral = DihedralAngle((O1, C3, O2, HO))

# DIHEDRALS - IMPROPER
C2O1C3O2_dihedral = DihedralAngle(atoms=[C2, O1, C3, O2], improper=True)

# DISPERSION
C1C1_disp = Dispersion(universe, (1, 1), cutoff = 8.0, vdw_tail_correction=True)
C2C2_disp = Dispersion(universe, (2, 2), cutoff = 8.0, vdw_tail_correction=True)
C3C3_disp = Dispersion(universe, (3, 3), cutoff = 8.0, vdw_tail_correction=True)
H1XH1X_disp = Dispersion(universe, (4, 4), cutoff = 8.0, vdw_tail_correction=True)
H1YH1Y_disp = Dispersion(universe, (5, 5), cutoff = 8.0, vdw_tail_correction=True)
H1ZH1Z_disp = Dispersion(universe, (6, 6), cutoff = 8.0, vdw_tail_correction=True)
H2XH2X_disp = Dispersion(universe, (7, 7), cutoff = 8.0, vdw_tail_correction=True)
H2YH2Y_disp = Dispersion(universe, (8, 8), cutoff = 8.0, vdw_tail_correction=True)
O1O1_disp = Dispersion(universe, (9, 9), cutoff = 8.0, vdw_tail_correction=True)
O2O2_disp = Dispersion(universe, (10, 10), cutoff = 8.0, vdw_tail_correction=True)
HOHO_disp = Dispersion(universe, (11, 11), cutoff = 8.0, vdw_tail_correction=True)















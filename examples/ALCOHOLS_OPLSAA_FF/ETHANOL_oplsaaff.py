from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined

C1 = Atom('C', position = [-4.914, 1.802, 0.137],  name = '80', charge = -0.042, atom_type = 1) 
C2 = Atom('C', position = [-3.588, 1.243, -0.406], name = '80', charge = 0.041,  atom_type = 2)   
H1X = Atom('H', position = [-4.728, 2.627, 0.828],  name = '81', charge = 0.025,  atom_type = 3)
H1Y = Atom('H', position = [-5.532, 2.173, -0.684], name = '81', charge = 0.025,  atom_type = 4)  
H1Z = Atom('H', position = [-5.472, 1.025, 0.664],  name = '81', charge = 0.025,  atom_type = 5)  
H2X = Atom('H', position = [-3.792, 0.424, -1.101], name = '81', charge = 0.055,  atom_type = 6)
H2Y = Atom('H', position = [-3.046, 2.032, -0.935], name = '81', charge = 0.055,  atom_type = 7)
O = Atom('O', position = [-2.794, 0.764, 0.680],  name = '96', charge = -0.395, atom_type =  8)
HO = Atom('H', position = [-1.959, 0.413, 0.321],  name = '97', charge = 0.211,  atom_type = 9)

# BONDS
C1H1X_bond = Bond(C1, H1X)
C1H1Y_bond = Bond(C1, H1Y)
C1H1Z_bond = Bond(C1, H1Z)
C1C2_bond = Bond(C1, C2)
C2H2X_bond = Bond(C2, H2X)
C2H2Y_bond = Bond(C2, H2Y)
C2O_bond = Bond(C2, O)
OHO_bond = Bond(O, HO, constrained=True)

# ANGLES
C2C1H1X_angle = BondAngle((C2, C1, H1X))
C2C1H1Y_angle = BondAngle((C2, C1, H1Y))
C2C1H1Z_angle = BondAngle((C2, C1, H1Z))
H1XC1H1Y_angle = BondAngle((H1X, C1, H1Y))
H1XC1H1Z_angle = BondAngle((H1X, C1, H1Z))
H1YC1H1Z_angle = BondAngle((H1Y, C1, H1Z))
C1C2H2X_angle = BondAngle((C1, C2, H2X))
C1C2H2Y_angle = BondAngle((C1, C2, H2Y))
C1C2O_angle = BondAngle((C1, C2, O))
H2XC2H2Y_angle =  BondAngle((H2X, C2, H2Y))
H2XC2O_angle = BondAngle((H2X, C2, O)) 
H2YC2O_angle = BondAngle((H2Y, C2, O)) 
HOOC2_angle = BondAngle((HO, O, C2)) 

# DIHEDRALS
H1XC1C2H2X_dihedral = DihedralAngle((H1X, C1, C2, H2X))
H1XC1C2H2Y_dihedral = DihedralAngle((H1X, C1, C2, H2Y))
H1XC1C2O_dihedral = DihedralAngle((H1X, C1, C2, O))
H1YC1C2H2X_dihedral = DihedralAngle((H1Y, C1, C2, H2X))
H1YC1C2H2Y_dihedral = DihedralAngle((H1Y, C1, C2, H2Y))
H1YC1C2O_dihedral = DihedralAngle((H1Y, C1, C2, O))
H1ZC1C2H2X_dihedral = DihedralAngle((H1Z, C1, C2, H2X))
H1ZC1C2H2Y_dihedral = DihedralAngle((H1Z, C1, C2, H2Y))
H1ZC1C2O_dihedral = DihedralAngle((H1Z, C1, C2, O))
C1C2OHO_dihedral = DihedralAngle((C1, C2, O, HO))
H2XC2OHO_dihedral = DihedralAngle((H2X, C2, O, HO))
H2YC2OHO_dihedral = DihedralAngle((H2Y, C2, O, HO))

# DISPERSION 
C1C1_disp = Dispersion(universe, (1, 1), cutoff = 8.0, vdw_tail_correction=True)
C2C2_disp = Dispersion(universe, (2, 2), cutoff = 8.0, vdw_tail_correction=True)
H1XH1X_disp = Dispersion(universe, (3, 3), cutoff = 8.0, vdw_tail_correction=True)
H1YH1Y_disp = Dispersion(universe, (4, 4), cutoff = 8.0, vdw_tail_correction=True)
H1ZH1Z_disp = Dispersion(universe, (5, 5), cutoff = 8.0, vdw_tail_correction=True)
H2XH2X_disp = Dispersion(universe, (6, 6), cutoff = 8.0, vdw_tail_correction=True)
H2YH2Y_disp = Dispersion(universe, (7, 7), cutoff = 8.0, vdw_tail_correction=True)
OO_disp = Dispersion(universe, (8, 8), cutoff = 8.0, vdw_tail_correction=True)
HOHO_disp = Dispersion(universe, (8, 8), cutoff = 8.0, vdw_tail_correction=True)

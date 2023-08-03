from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined

C1  = Atom('C', position = [ 2.680,  -0.350,  -0.320], name = '853',   charge =  0.359, atom_type = 1) 
C2  = Atom('C', position = [ 1.270,  -0.940,  -0.560], name = ' 80',  charge =  0.089, atom_type = 2)
C3  = Atom('C', position = [-0.000,  -0.300,   0.170], name = ' 80',  charge =  0.186, atom_type = 3)
C4  = Atom('C', position = [-1.330,  -0.910,  -0.500], name = ' 80',  charge =  0.089, atom_type = 4)
C5  = Atom('C', position = [-2.730,  -0.320,  -0.180], name = '853',  charge =  0.359, atom_type = 5)
C6  = Atom('C', position = [ 0.120,   1.250,   0.260], name = '853',  charge =  0.388, atom_type = 6)
O1  = Atom('O', position = [ 3.630,  -1.220,   0.030], name = '854',  charge = -0.246, atom_type = 7) 
O2  = Atom('O', position = [ 2.970,   0.760,  -0.720], name = '854',    charge = -0.246, atom_type = 8)
O3  = Atom('O', position = [ 0.020,  -0.760,   1.540], name = ' 96',   charge = -0.376, atom_type = 9) 
O4  = Atom('O', position = [-0.860,   1.990,  -0.220], name = '854',  charge = -0.243, atom_type = 10)
O5  = Atom('O', position = [ 1.010,   1.770,   0.910], name = '854',  charge = -0.243, atom_type = 11)
O6  = Atom('O', position = [-3.660,  -1.200,   0.180], name = '854',  charge = -0.246, atom_type = 12)
O7  = Atom('O', position = [-3.100,   0.760,  -0.600], name = '854',  charge = -0.246, atom_type = 13)
H2X = Atom('H', position = [ 1.300,  -2.020,  -0.350], name = ' 85',  charge =  0.041, atom_type = 14)
H2Y = Atom('H', position = [ 1.100,  -0.870,  -1.640], name = ' 85',  charge =  0.041, atom_type = 15) 
H2O = Atom('H', position = [ 0.010,  -1.730,   1.560], name = ' 97',    charge =  0.212, atom_type = 16)
H4X = Atom('H', position = [-1.360,  -1.980,  -0.240], name = ' 85',  charge =  0.041, atom_type = 17) 
H4Y = Atom('H', position = [-1.200,  -0.890,  -1.580], name = ' 85',    charge =  0.041, atom_type = 18)

# BONDS
C1C2_bond = Bond(C1, C2)
C1O1_bond = Bond(C1, O1)
C1O2_bond = Bond(C1, O2)
C2C3_bond = Bond(C2, C3)
C2H2X_bond = Bond(C2, H2X)
C2H2Y_bond = Bond(C2, H2Y)
C3C4_bond = Bond(C3, C4)
C3C6_bond = Bond(C3, C6)
C3O3_bond = Bond(C3, O3)
C4C5_bond = Bond(C4, C5)
C4H4X_bond = Bond(C4, H4X)
C4H4Y_bond = Bond(C4, H4Y)
C5O6_bond = Bond(C5, O6)
C5O7_bond = Bond(C5, O7)
C6O4_bond = Bond(C6, O4)
C6O5_bond = Bond(C6, O5)
O3H2O_bond = Bond(O3, H2O)

# ANGLES
C1C2C3_angle = BondAngle((C1, C2, C3))
C1C2H2X_angle = BondAngle((C1, C2, H2X))
C1C2H2Y_angle = BondAngle((C1, C2, H2Y))
C2C1O1_angle = BondAngle((C2, C1, O1))
C2C1O2_angle = BondAngle((C2, C1, O2))
C2C3C4_angle = BondAngle((C2, C3, C4))
C2C3C6_angle = BondAngle((C2, C3, C6))
C2C3O3_angle = BondAngle((C2, C3, O3))
C3C2H2X_angle = BondAngle((C3, C2, H2X))
C3C2H2Y_angle = BondAngle((C3, C2, H2Y))
C3C4C5_angle = BondAngle((C3, C4, C5))
C3C4H4X_angle = BondAngle((C3, C4, H4X))
C3C4H4Y_angle = BondAngle((C3, C4, H4Y))
C3C6O4_angle = BondAngle((C3, C6, O4))
C3C6O5_angle = BondAngle((C3, C6, O5))
C3O3H2O_angle = BondAngle((C3, O3, H2O))
C4C3C6_angle = BondAngle((C4, C3, C6))
C4C3O3_angle = BondAngle((C4, C3, O3))
C4C5O6_angle = BondAngle((C4, C5, O6))
C4C5O7_angle = BondAngle((C4, C5, O7))
C5C4H4X_angle = BondAngle((C5, C4, H4X))
C5C4H4Y_angle = BondAngle((C5, C4, H4Y))
C6C3O3_angle = BondAngle((C6, C3, O3))
O1C1O2_angle = BondAngle((O1, C1, O2))
O4C6O5_angle = BondAngle((O4, C6, O5))
O6C5O7_angle = BondAngle((O6, C5, O7))
H2XC2H2Y_angle = BondAngle((H2X, C2, H2Y))
H4XC4H4Y_angle = BondAngle((H4X, C4, H4Y))

# DIHEDRALS - PROPER
C1C2C3C4_dihedral = DihedralAngle((C1, C2, C3, C4))
C1C2C3C6_dihedral = DihedralAngle((C1, C2, C3, C6))
C1C2C3O3_dihedral = DihedralAngle((C1, C2, C3, O3))
C2C3C4C5_dihedral = DihedralAngle((C2, C3, C4, C5))
C2C3C4H4X_dihedral = DihedralAngle((C2, C3, C4, H4X))
C2C3C4H4Y_dihedral = DihedralAngle((C2, C3, C4, H4Y))
C2C3C6O4_dihedral = DihedralAngle((C2, C3, C6, O4))
C2C3C6O5_dihedral = DihedralAngle((C2, C3, C6, O5))
C2C3O3H2O_dihedral = DihedralAngle((C2, C3, O3, H2O))
C3C4C5O6_dihedral = DihedralAngle((C3, C4, C5, O6))
C3C4C5O7_dihedral = DihedralAngle((C3, C4, C5, O7))
C4C3C2H2X_dihedral = DihedralAngle((C4, C3, C2, H2X))
C4C3C2H2Y_dihedral = DihedralAngle((C4, C3, C2, H2Y))
C4C3C6O4_dihedral = DihedralAngle((C4, C3, C6, O4))
C4C3C6O5_dihedral = DihedralAngle((C4, C3, C6, O5))
C4C3O3H2O_dihedral = DihedralAngle((C4, C3, O3, H2O))
C5C4C3C6_dihedral = DihedralAngle((C5, C4, C3, C6))
C5C4C3O3_dihedral = DihedralAngle((C5, C4, C3, O3))
C6C3C2H2X_dihedral = DihedralAngle((C6, C3, C2, H2X))
C6C3C2H2Y_dihedral = DihedralAngle((C6, C3, C2, H2Y))
C6C3C4H4X_dihedral = DihedralAngle((C6, C3, C4, H4X))
C6C3C4H4Y_dihedral = DihedralAngle((C6, C3, C4, H4Y))
C6C3O3H2O_dihedral = DihedralAngle((C6, C3, O3, H2O))
O1C1C2C3_dihedral = DihedralAngle((O1, C1, C2, C3))
O1C1C2H2X_dihedral = DihedralAngle((O1, C1, C2, H2X))
O1C1C2H2Y_dihedral = DihedralAngle((O1, C1, C2, H2Y))
O2C1C2C3_dihedral = DihedralAngle((O2, C1, C2, C3))
O2C1C2H2X_dihedral = DihedralAngle((O2, C1, C2, H2X))
O2C1C2H2Y_dihedral = DihedralAngle((O2, C1, C2, H2Y))
O3C3C2H2X_dihedral = DihedralAngle((O3, C3, C2, H2X))
O3C3C2H2Y_dihedral = DihedralAngle((O3, C3, C2, H2Y))
O3C3C4H4X_dihedral = DihedralAngle((O3, C3, C4, H4X))
O3C3C4H4Y_dihedral = DihedralAngle((O3, C3, C4, H4Y))
O3C3C6O4_dihedral = DihedralAngle((O3, C3, C6, O4))
O3C3C6O5_dihedral = DihedralAngle((O3, C3, C6, O5))
O6C5C4H4X_dihedral = DihedralAngle((O6, C5, C4, H4X))
O6C5C4H4Y_dihedral = DihedralAngle((O6, C5, C4, H4Y))
O7C5C4H4X_dihedral = DihedralAngle((O7, C5, C4, H4X))
O7C5C4H4Y_dihedral = DihedralAngle((O7, C5, C4, H4Y))

# DIHEDRALS - IMPROPER
C3O4C6O5_dihedral = DihedralAngle(atoms=[C3, O4, C6, O5],  improper=True)
C4O6C5O7_dihedral = DihedralAngle(atoms=[C4, O6, C5, O7],  improper=True)
O2C1O1C2_dihedral = DihedralAngle(atoms=[O2, C1, O1, C2],  improper=True)

# DISPERSION
C1C1_disp = Dispersion(universe, (1,1), cutoff = 8.0, vdw_tail_correction=True)
C2C2_disp = Dispersion(universe, (2,2), cutoff = 8.0, vdw_tail_correction=True)
C3C3_disp = Dispersion(universe, (3,3), cutoff = 8.0, vdw_tail_correction=True)
C4C4_disp = Dispersion(universe, (4,4), cutoff = 8.0, vdw_tail_correction=True)
C5C5_disp = Dispersion(universe, (5,5), cutoff = 8.0, vdw_tail_correction=True)
C6C6_disp = Dispersion(universe, (6,6), cutoff = 8.0, vdw_tail_correction=True)
O1O1_disp = Dispersion(universe, (7,7), cutoff = 8.0, vdw_tail_correction=True)
O2O2_disp = Dispersion(universe, (8,8), cutoff = 8.0, vdw_tail_correction=True)
O3O3_disp = Dispersion(universe, (9,9), cutoff = 8.0, vdw_tail_correction=True)
O4O4_disp = Dispersion(universe, (10,10), cutoff = 8.0, vdw_tail_correction=True)
O5O5_disp = Dispersion(universe, (11,11), cutoff = 8.0, vdw_tail_correction=True)
O6O6_disp = Dispersion(universe, (12,12), cutoff = 8.0, vdw_tail_correction=True)
O7O7_disp = Dispersion(universe, (13,13), cutoff = 8.0, vdw_tail_correction=True)
H2XH2X_disp = Dispersion(universe, (14,14), cutoff = 8.0, vdw_tail_correction=True)
H2YH2Y_disp = Dispersion(universe, (15,15), cutoff = 8.0, vdw_tail_correction=True)
H2OH2O_disp = Dispersion(universe, (16,16), cutoff = 8.0, vdw_tail_correction=True)
H4XH4X_disp = Dispersion(universe, (17,17), cutoff = 8.0, vdw_tail_correction=True)
H4YH4Y_disp = Dispersion(universe, (18,18), cutoff = 8.0, vdw_tail_correction=True)
















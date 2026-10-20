from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined

C1  = Atom('C', position = [-4.298,   1.136,  -0.111], name = ' 80',   charge = -0.150, atom_type = 1) 
C3  = Atom('C', position = [-2.832,   1.023,   0.054], name = '853',  charge =  0.636, atom_type = 2)
H1X = Atom('H', position = [-4.602,   2.175,  -0.293], name = ' 85',  charge =  0.078, atom_type = 3)
H1Y = Atom('H', position = [-4.792,   0.752,   0.803], name = ' 85',  charge =  0.078, atom_type = 4)
H1Z = Atom('H', position = [-4.618,   0.501,  -0.963], name = ' 85',  charge =  0.078, atom_type = 5)
O1  = Atom('O', position = [-2.108,   2.017,   0.112], name = '854',  charge = -0.552, atom_type = 6)
O2  = Atom('O', position = [-2.289,  -0.182,   0.147], name = ' 96',  charge = -0.611, atom_type = 7) 
HO  = Atom('H', position = [-1.340,  -0.216,   0.251], name = ' 97',    charge =  0.444, atom_type = 8)

# BONDS
C1C3_bond = Bond(C1,C3)
C1H1X_bond = Bond(C1,H1X)
C1H1Y_bond = Bond(C1,H1Y)
C1H1Z_bond = Bond(C1,H1Z)
C3O1_bond = Bond(C3,O1)
C3O2_bond = Bond(C3,O2)
O2HO_bond = Bond(O2,HO)

# ANGLES
C1C3O1_angle = BondAngle((C1,C3,O1))
C1C3O2_angle = BondAngle((C1,C3,O2))
C3C1H1X_angle = BondAngle((C3,C1,H1X))
C3C1H1Y_angle = BondAngle((C3,C1,H1Y))
C3C1H1Z_angle = BondAngle((C3,C1,H1Z))
C3O2HO_angle = BondAngle((C3,O2,HO))
H1XC1H1Y_angle = BondAngle((H1X,C1,H1Y))
H1XC1H1Z_angle = BondAngle((H1X,C1,H1Z))
H1YC1H1Z_angle = BondAngle((H1Y,C1,H1Z))
O1C3O2_angle = BondAngle((O1,C3,O2))

# DIHEDRALS
C1C3O2HO_dihedral = DihedralAngle((C1,C3,O2,HO))
H1XC1C3O1_dihedral = DihedralAngle((H1X,C1,C3,O1))
H1XC1C3O2_dihedral = DihedralAngle((H1X,C1,C3,O2))
H1YC1C3O1_dihedral = DihedralAngle((H1Y,C1,C3,O1))
H1YC1C3O2_dihedral = DihedralAngle((H1Y,C1,C3,O2))
H1ZC1C3O1_dihedral = DihedralAngle((H1Z,C1,C3,O1))
H1ZC1C3O2_dihedral = DihedralAngle((H1Z,C1,C3,O2))
O1C3O2HO_dihedral = DihedralAngle((O1,C3,O2,HO))

# DISPERSION
C1C1_disp = Dispersion(universe, (1,1), cutoff = 8.0, vdw_tail_correction=True)
C3C3_disp = Dispersion(universe, (2,2), cutoff = 8.0, vdw_tail_correction=True)
H1XH1X_disp = Dispersion(universe, (3,3), cutoff = 8.0, vdw_tail_correction=True)
H1YH1Y_disp = Dispersion(universe, (4,4), cutoff = 8.0, vdw_tail_correction=True)
H1ZH1Z_disp = Dispersion(universe, (5,5), cutoff = 8.0, vdw_tail_correction=True)
O1O1_disp = Dispersion(universe, (6,6), cutoff = 8.0, vdw_tail_correction=True)
O2O2_disp = Dispersion(universe, (7,7), cutoff = 8.0, vdw_tail_correction=True)
HOHO_disp = Dispersion(universe, (8,8), cutoff = 8.0, vdw_tail_correction=True)




























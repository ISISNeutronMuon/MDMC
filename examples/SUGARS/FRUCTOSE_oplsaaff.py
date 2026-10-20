from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined


C1  = Atom('C', position = [-1.915,  -1.718,   0.224], name = '23', charge =  0.298, atom_type = 1) 
C2  = Atom('C', position = [-1.205,  -0.519,   0.835], name = '761', charge = -0.157, atom_type = 2)
C3  = Atom('C', position = [-1.340,   0.783,   0.037], name = '761', charge =  0.555, atom_type = 3)
C4  = Atom('C', position = [ 0.081,   1.339,   0.024], name = '761', charge =  0.117, atom_type = 4)
C5  = Atom('C', position = [ 0.938,   0.058,   0.053], name = '761', charge =  0.490, atom_type = 5)
C6  = Atom('C', position = [ 2.344,   0.261,   0.612], name = '23', charge =  0.099, atom_type = 6)
O1  = Atom('O', position = [ 0.218,  -0.786,   0.932], name = '507', charge = -0.416, atom_type = 7) 
O2  = Atom('O', position = [-1.626,  -1.913,  -1.147], name = '5', charge = -0.722, atom_type = 8)
O3  = Atom('O', position = [-2.289,   1.631,   0.666], name = '5', charge = -0.775, atom_type = 9)
O4  = Atom('O', position = [ 0.249,   2.203,  -1.079], name = '5', charge = -0.715, atom_type = 10)
O5  = Atom('O', position = [ 1.018,  -0.494,  -1.243], name = '5', charge = -0.711, atom_type = 11)
O6  = Atom('O', position = [ 3.103,  -0.907,   0.316], name = '5', charge = -0.713, atom_type = 12)
H11C= Atom('H', position = [-2.994,  -1.567,   0.304], name = '85', charge =  0.035, atom_type = 13) 
H12C= Atom('H', position = [-1.655,  -2.604,   0.815], name = '85', charge =  0.017, atom_type = 14)
H2C = Atom('H', position = [-1.577,  -0.359,   1.849], name = '510', charge =  0.133, atom_type = 15)
H3C = Atom('H', position = [-1.633,   0.569,  -0.992], name = '510', charge =  0.015, atom_type = 16)
H4C = Atom('H', position = [ 0.244,   1.867,   0.971], name = '510', charge =  0.034, atom_type = 17)
H61C= Atom('H', position = [ 2.277,   0.431,   1.689], name = '85', charge =  0.055, atom_type = 18)
H62C= Atom('H', position = [ 2.789,   1.140,   0.134], name = '85', charge =  0.044, atom_type = 19) 
H2O = Atom('H', position = [-0.682,  -1.749,  -1.288], name = '765', charge =  0.454, atom_type = 20)
H3O = Atom('H', position = [-2.499,   2.345,   0.057], name = '765', charge =  0.453, atom_type = 21)
H4O = Atom('H', position = [ 1.103,   2.641,  -1.024], name = '765', charge =  0.461, atom_type = 22)
H5O = Atom('H', position = [ 1.783,  -1.088,  -1.233], name = '765', charge =  0.471, atom_type = 23)
H6O = Atom('H', position = [ 4.038,  -0.723,   0.437], name = '765', charge =  0.478, atom_type = 24)


# BONDS
C1C2_bond = Bond(C1, C2)
C1O2_bond = Bond(C1, O2)
C1H11C_bond = Bond(C1, H11C, constrained=True)
C1H12C_bond = Bond(C1, H12C, constrained=True)
C2C3_bond = Bond(C2, C3)
C2O1_bond = Bond(C2, O1)
C2H2C_bond = Bond(C2, H2C, constrained=True)
C3C4_bond = Bond(C3, C4)
C3O3_bond = Bond(C3, O3)
C3H3C_bond = Bond(C3, H3C, constrained=True)
C4C5_bond = Bond(C4, C5)
C4O4_bond = Bond(C4, O4)
C4H4C_bond = Bond(C4, H4C, constrained=True)
C5C6_bond = Bond(C5, C6)
C5O1_bond = Bond(C5, O1)
C5O5_bond = Bond(C5, O5)
C6O6_bond = Bond(C6, O6)
C6H61C_bond = Bond(C6, H61C, constrained=True)
C6H62C_bond = Bond(C6, H62C, constrained=True)
O2H2O_bond = Bond(O2, H2O, constrained=True)
O3H3O_bond = Bond(O3, H3O, constrained=True)
O4H4O_bond = Bond(O4, H4O, constrained=True)
O5H5O_bond = Bond(O5, H5O, constrained=True)
O6H6O_bond = Bond(O6, H6O, constrained=True)

# ANGLES
C1C2C3_angle = BondAngle((C1, C2, C3))
C1C2O1_angle = BondAngle((C1, C2, O1))
C1C2H2C_angle = BondAngle((C1, C2, H2C))
C1O2H2O_angle = BondAngle((C1, O2, H2O))
C2C1O2_angle = BondAngle((C2, C1, O2))
C2C1H11C_angle = BondAngle((C2, C1, H11C))
C2C1H12C_angle = BondAngle((C2, C1, H12C))
C2C3C4_angle = BondAngle((C2, C3, C4))
C2C3O3_angle = BondAngle((C2, C3, O3))
C2C3H3C_angle = BondAngle((C2, C3, H3C))
C2O1C5_angle = BondAngle((C2, O1, C5))
C3C2O1_angle = BondAngle((C3, C2, O1))
C3C2H2C_angle = BondAngle((C3, C2, H2C))
C3C4C5_angle = BondAngle((C3, C4, C5))
C3C4O4_angle = BondAngle((C3, C4, O4))
C3C4H4C_angle = BondAngle((C3, C4, H4C))
C3O3H3O_angle = BondAngle((C3, O3, H3O))
C4C3O3_angle = BondAngle((C4, C3, O3))
C4C3H3C_angle = BondAngle((C4, C3, H3C))
C4C5C6_angle = BondAngle((C4, C5, C6))
C4C5O1_angle = BondAngle((C4, C5, O1))
C4C5O5_angle = BondAngle((C4, C5, O5))
C4O4H4O_angle = BondAngle((C4, O4, H4O))
C5C4O4_angle = BondAngle((C5, C4, O4))
C5C4H4C_angle = BondAngle((C5, C4, H4C))
C5C6O6_angle = BondAngle((C5, C6, O6))
C5C6H61C_angle = BondAngle((C5, C6, H61C))
C5C6H62C_angle = BondAngle((C5, C6, H62C))
C5O5H5O_angle = BondAngle((C5, O5, H5O))
C6C5O1_angle = BondAngle((C6, C5, O1))
C6C5O5_angle = BondAngle((C6, C5, O5))
C6O6H6O_angle = BondAngle((C6, O6, H6O))
O1C2H2C_angle = BondAngle((O1, C2, H2C))
O1C5O5_angle = BondAngle((O1, C5, O5))
O2C1H11C_angle = BondAngle((O2, C1, H11C))
O2C1H12C_angle = BondAngle((O2, C1, H12C))
O3C3H3C_angle = BondAngle((O3, C3, H3C))
O4C4H4C_angle = BondAngle((O4, C4, H4C))
O6C6H61C_angle = BondAngle((O6, C6, H61C))
O6C6H62C_angle = BondAngle((O6, C6, H62C))
H11CC1H12C_angle = BondAngle((H11C, C1, H12C))
H61CC6H62C_angle = BondAngle((H61C, C6, H62C))

# DIHEDRALS
C1C2C3C4_dihedral = DihedralAngle((C1, C2, C3, C4))
C1C2C3O3_dihedral = DihedralAngle((C1, C2, C3, O3))
C1C2C3H3C_dihedral = DihedralAngle((C1, C2, C3, H3C))
C1C2O1C5_dihedral = DihedralAngle((C1, C2, O1, C5))
C2C1O2H2O_dihedral = DihedralAngle((C2, C1, O2, H2O))
C2C3C4C5_dihedral = DihedralAngle((C2, C3, C4, C5))
C2C3C4O4_dihedral = DihedralAngle((C2, C3, C4, O4))
C2C3C4H4C_dihedral = DihedralAngle((C2, C3, C4, H4C))
C2C3O3H3O_dihedral = DihedralAngle((C2, C3, O3, H3O))
C2O1C5C4_dihedral = DihedralAngle((C2, O1, C5, C4))
C2O1C5C6_dihedral = DihedralAngle((C2, O1, C5, C6))
C2O1C5O5_dihedral = DihedralAngle((C2, O1, C5, O5))
C3C2O1C5_dihedral = DihedralAngle((C3, C2, O1, C5))
C3C4C5C6_dihedral = DihedralAngle((C3, C4, C5, C6))
C3C4C5O1_dihedral = DihedralAngle((C3, C4, C5, O1))
C3C4C5O5_dihedral = DihedralAngle((C3, C4, C5, O5))
C3C4O4H4O_dihedral = DihedralAngle((C3, C4, O4, H4O))
C4C3C2O1_dihedral = DihedralAngle((C4, C3, C2, O1))
C4C3C2H2C_dihedral = DihedralAngle((C4, C3, C2, H2C))
C4C3O3H3O_dihedral = DihedralAngle((C4, C3, O3, H3O))
C4C5C6O6_dihedral = DihedralAngle((C4, C5, C6, O6))
C4C5C6H61C_dihedral = DihedralAngle((C4, C5, C6, H61C))
C4C5C6H62C_dihedral = DihedralAngle((C4, C5, C6, H62C))
C4C5O5H5O_dihedral = DihedralAngle((C4, C5, O5, H5O))
C5C4C3O3_dihedral = DihedralAngle((C5, C4, C3, O3))
C5C4C3H3C_dihedral = DihedralAngle((C5, C4, C3, H3C))
C5C4O4H4O_dihedral = DihedralAngle((C5, C4, O4, H4O))
C5C6O6H6O_dihedral = DihedralAngle((C5, C6, O6, H6O))
C5O1C2H2C_dihedral = DihedralAngle((C5, O1, C2, H2C))
C6C5C4O4_dihedral = DihedralAngle((C6, C5, C4, O4))
C6C5C4H4C_dihedral = DihedralAngle((C6, C5, C4, H4C))
C6C5O5H5O_dihedral = DihedralAngle((C6, C5, O5, H5O))
O1C2C3O3_dihedral = DihedralAngle((O1, C2, C3, O3))
O1C2C3H3C_dihedral = DihedralAngle((O1, C2, C3, H3C))
O1C5C4O4_dihedral = DihedralAngle((O1, C5, C4, O4))
O1C5C4H4C_dihedral = DihedralAngle((O1, C5, C4, H4C))
O1C5C6O6_dihedral = DihedralAngle((O1, C5, C6, O6))
O1C5C6H61C_dihedral = DihedralAngle((O1, C5, C6, H61C))
O1C5C6H62C_dihedral = DihedralAngle((O1, C5, C6, H62C))
O1C5O5H5O_dihedral = DihedralAngle((O1, C5, O5, H5O))
O2C1C2C3_dihedral = DihedralAngle((O2, C1, C2, C3))
O2C1C2O1_dihedral = DihedralAngle((O2, C1, C2, O1))
O2C1C2H2C_dihedral = DihedralAngle((O2, C1, C2, H2C))
O3C3C2H2C_dihedral = DihedralAngle((O3, C3, C2, H2C))
O3C3C4O4_dihedral = DihedralAngle((O3, C3, C4, O4))
O3C3C4H4C_dihedral = DihedralAngle((O3, C3, C4, H4C))
O4C4C3H3C_dihedral = DihedralAngle((O4, C4, C3, H3C))
O4C4C5O5_dihedral = DihedralAngle((O4, C4, C5, O5))
O5C5C4H4C_dihedral = DihedralAngle((O5, C5, C4, H4C))
O5C5C6O6_dihedral = DihedralAngle((O5, C5, C6, O6))
O5C5C6H61C_dihedral = DihedralAngle((O5, C5, C6, H61C))
O5C5C6H62C_dihedral = DihedralAngle((O5, C5, C6, H62C))
H11CC1C2C3_dihedral = DihedralAngle((H11C, C1, C2, C3))
H11CC1C2O1_dihedral = DihedralAngle((H11C, C1, C2, O1))
H11CC1C2H2C_dihedral = DihedralAngle((H11C, C1, C2, H2C))
H11CC1O2H2O_dihedral = DihedralAngle((H11C, C1, O2, H2O))
H12CC1C2C3_dihedral = DihedralAngle((H12C, C1, C2, C3))
H12CC1C2O1_dihedral = DihedralAngle((H12C, C1, C2, O1))
H12CC1C2H2C_dihedral = DihedralAngle((H12C, C1, C2, H2C))
H12CC1O2H2O_dihedral = DihedralAngle((H12C, C1, O2, H2O))
H2CC2C3H3C_dihedral = DihedralAngle((H2C, C2, C3, H3C))
H3CC3C4H4C_dihedral = DihedralAngle((H3C, C3, C4, H4C))
H3CC3O3H3O_dihedral = DihedralAngle((H3C, C3, O3, H3O))
H4CC4O4H4O_dihedral = DihedralAngle((H4C, C4, O4, H4O))
H61CC6O6H6O_dihedral = DihedralAngle((H61C, C6, O6, H6O))
H62CC6O6H6O_dihedral = DihedralAngle((H62C, C6, O6, H6O))

# DISPERSION
C1C1_disp = Dispersion(universe, (1, 1), cutoff = 8.0, vdw_tail_correction=True)
C2C2_disp = Dispersion(universe, (2, 2), cutoff = 8.0, vdw_tail_correction=True)
C3C3_disp = Dispersion(universe, (3, 3), cutoff = 8.0, vdw_tail_correction=True)
C4C4_disp = Dispersion(universe, (4, 4), cutoff = 8.0, vdw_tail_correction=True)
C5C5_disp = Dispersion(universe, (5, 5), cutoff = 8.0, vdw_tail_correction=True)
C6C6_disp = Dispersion(universe, (6, 6), cutoff = 8.0, vdw_tail_correction=True)
O1O1_disp = Dispersion(universe, (7, 7), cutoff = 8.0, vdw_tail_correction=True)
O2O2_disp = Dispersion(universe, (8, 8), cutoff = 8.0, vdw_tail_correction=True)
O3O3_disp = Dispersion(universe, (9, 9), cutoff = 8.0, vdw_tail_correction=True)
O4O4_disp = Dispersion(universe, (10, 10), cutoff = 8.0, vdw_tail_correction=True)
O5O5_disp = Dispersion(universe, (11, 11), cutoff = 8.0, vdw_tail_correction=True)
O6O6_disp = Dispersion(universe, (12, 12), cutoff = 8.0, vdw_tail_correction=True)
H11CH11C_disp = Dispersion(universe, (13, 13), cutoff = 8.0, vdw_tail_correction=True)
H12CH12C_disp = Dispersion(universe, (14, 14), cutoff = 8.0, vdw_tail_correction=True)
H2CH2C_disp = Dispersion(universe, (15, 15), cutoff = 8.0, vdw_tail_correction=True)
H3CH3C_disp = Dispersion(universe, (16, 16), cutoff = 8.0, vdw_tail_correction=True)
H4CH4C_disp = Dispersion(universe, (17, 17), cutoff = 8.0, vdw_tail_correction=True)
H61CH61C_disp = Dispersion(universe, (18, 18), cutoff = 8.0, vdw_tail_correction=True)
H62CH62C_disp = Dispersion(universe, (19, 19), cutoff = 8.0, vdw_tail_correction=True)
H2OH2O_disp = Dispersion(universe, (20, 20), cutoff = 8.0, vdw_tail_correction=True)
H3OH3O_disp = Dispersion(universe, (21, 21), cutoff = 8.0, vdw_tail_correction=True)
H4OH4O_disp = Dispersion(universe, (22, 22), cutoff = 8.0, vdw_tail_correction=True)
H5OH5O_disp = Dispersion(universe, (23, 23), cutoff = 8.0, vdw_tail_correction=True)
H6OH6O_disp = Dispersion(universe, (24, 24), cutoff = 8.0, vdw_tail_correction=True)
































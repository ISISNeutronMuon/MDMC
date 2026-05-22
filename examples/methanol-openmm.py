from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
HC1 = Atom("H", position=[-0.7006,  0.3636,  0.8900], name="98", atom_type="98")
HC2 = Atom("H", position=[-0.7006,  0.3636, -0.8900], name="98", atom_type="98")
HC3 = Atom("H", position=[-0.7076, -1.1754,  0.0000], name="98", atom_type="98")
C = Atom("C", position=[-0.3366, -0.1504,  0.0000], name="99", atom_type="99")
O = Atom("O", position=[ 1.0849, -0.1713,  0.0000], name="96", atom_type="96")
HO = Atom("H", position=[ 1.3606,  0.7699,  0.0000], name="97", atom_type="97")

# Create the methanol Molecule
methanol = Molecule(
    atoms=[HC1, HC2, HC3, C, O, HO],
    interactions=[
        Bond((C, HC1), (C, HC2), (C, HC3), (O, HO)),
        Bond((C, O), constrained=True),
        BondAngle(
            (HC1, C, O), (HC2, C, O), (HC3, C, O),
            (HC1, C, HC2), (HC2, C, HC3), (HC3, C, HC1),
            (HO, O, C)
        ),
        DihedralAngle((HC1, C, O, HO), (HC2, C, O, HO), (HC3, C, O, HO))
    ]
)

# Create a universe and add the methanol
universe = Universe(dimensions=15.0, constraint_algorithm=Shake(1e-5, 100), electrostatic_solver=PPPM(accuracy=1e-4))
universe.fill(methanol, num_density=0.01)

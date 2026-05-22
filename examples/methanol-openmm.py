import os

from MDMC.MD import *
from MDMC.MD.force_fields.OPLSAA import add_opls_force_field

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

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
        Bond((C, HC1), (C, HC2), (C, HC3)),
        Bond((O, HO)),
        Bond((C, O), constrained=True),
        BondAngle((HC1, C, O), (HC2, C, O), (HC3, C, O)),
        BondAngle((HC1, C, HC2), (HC2, C, HC3), (HC3, C, HC1)),
        BondAngle((HO, O, C)),
        DihedralAngle((HC1, C, O, HO), (HC2, C, O, HO), (HC3, C, O, HO))
    ]
)

# Create a universe and add the methanol
universe = Universe(dimensions=15.0, constraint_algorithm=Shake(1e-5, 100), electrostatic_solver=PPPM(accuracy=1e-4))
universe.fill(methanol, num_density=0.01)
add_opls_force_field(universe, cutoff=6.0, ewald=1e-4)

simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=300,
    traj_step=10,
    openmm_platform="OpenCL",
    # default precision on CUDA and OpenCL is single
    openmm_properties={"Precision": "single"},
)

simulation.run(n_steps=30000, equilibration=True)

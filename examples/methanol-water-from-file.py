import os

from MDMC.MD import *
from MDMC.MD.force_fields.OPLSAA import add_opls_force_field

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
H1 = Atom("H", position=[-0.7006, 0.3636, 0.8900], name="98", atom_type="98")
H2 = Atom("H", position=[-0.7006, 0.3636, -0.8900], name="98", atom_type="98")
H3 = Atom("H", position=[-0.7076, -1.1754, 0.0000], name="98", atom_type="98")
C = Atom("C", position=[-0.3366, -0.1504, 0.0000], name="99", atom_type="99")
O = Atom("O", position=[1.0849, -0.1713, 0.0000], name="96", atom_type="96")
H4 = Atom("H", position=[1.3606, 0.7699, 0.0000], name="97", atom_type="97")

# the water atom names match those in the PDB file.
OW = Atom("O", position=[1.0849, -0.1713, 0.0000], name="63", atom_type="63")
HW1 = Atom("H", position=[1.0849, -0.1713, 0.0000], name="64", atom_type="64")
HW2 = Atom("H", position=[1.0849, -0.1713, 0.0000], name="64", atom_type="64")

interactions = [
    Bond((C, H1), (C, H2), (C, H3)),
    Bond((O, H4)),
    Bond((C, O)),
    BondAngle((H1, C, O), (H2, C, O), (H3, C, O)),
    BondAngle((H1, C, H2), (H2, C, H3), (H3, C, H1)),
    BondAngle((H4, O, C)),
    DihedralAngle((H1, C, O, H4), (H2, C, O, H4), (H3, C, O, H4)),
    Bond((HW1, OW), (HW2, OW)),
    BondAngle((HW1, OW, HW2)),
]
# Create the universe from file
universe = Universe.from_pdb_file("structure/METHANOL_WATER.pdb")
add_opls_force_field(universe, cutoff=6.0, ewald=1e-4)

simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=300,
    traj_step=10,
    openmm_platform="OpenCL",
    # below needed since we are using OPLS
    openmm_nonbonded_scaling=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 0.5],
    ],
    openmm_nonbonded_combining="GEOMETRIC",
)

simulation.run(n_steps=30000, equilibration=True)

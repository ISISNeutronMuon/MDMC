import os

from MDMC.MD import *
from MDMC.readers import configurations
from MDMC.MD.force_fields.OPLSAA import add_opls_force_field

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

# loading molecule from cif file and set atom names to the OPLS atom types
atoms = configurations.read("Paracetamol.cif", name=[
    "109", "178", # Oxygens
    "207", # Nitrogen
    "208", "108", "90", "177", "90", "90", "90", "185", # Carbons
    "85", "85", "85", "91", "91", "91", "91", "183", "110"  # Hydrogens
])
paracetamol = Molecule(atoms=atoms)

# Create a universe and add the paracetamol molecules
universe = Universe(dimensions=15.0)
universe.fill(paracetamol, num_density=0.01)
add_opls_force_field(universe, cutoff=6.0, ewald=1e-4)

simulation = Simulation(
    universe,
    engine="openmm",
    time_step=1.0,
    temperature=300,
    traj_step=10,
    openmm_platform="OpenCL",
    # default precision on CUDA and OpenCL is single
    openmm_properties={"Precision": "mixed"},
    # below needed since we are using OPLS
    openmm_nonbonded_scaling=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 1.0, 0.5],
    ],
    openmm_nonbonded_combining="GEOMETRIC"
)

simulation.run(n_steps=30000, equilibration=True)

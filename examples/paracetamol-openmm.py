import os

from MDMC.MD import *
from MDMC.readers import configurations
from MDMC.MD.force_fields.OPLSAA import add_opls_force_field

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

# loading molecule from cif file and set atom names to the OPLS atom types
# so parameters can be set
atoms = configurations.read("Paracetamol.cif", name=[
    "109", "178", # Oxygens
    "207", # Nitrogen
    "208", "108", "90", "177", "90", "90", "90", "80", # Carbons
    "85", "85", "85", "91", "91", "91", "91", "183", "110"  # Hydrogens
])
paracetamol = Molecule(atoms=atoms)

# set the same labels so it's the same as libpargen
# for easy comparison using CC(=O)Nc1ccc(O)cc1 as input and openmm xml
# as output
(
    O808, # Phenol -OH O
    O802, # Amide C=O oxygen
    N803, # N-Phenylacetamide N
    C804, # N-Phenylacetamide N-CA C
    C807, # Phenol C-OH C
    C810, # Aromatic C
    C801, # Amide C=O C
    C805, C806, C809, # Aromatic C
    C800, # Alkane CH3- (note sure if this is the best type)
    H811, H812, H813,  # Alkane H-C H
    H815, H818, H819, H816, # Aromatic H-C H
    H814, # Amide -CO-NHR H
    H817  # Phenol -OH H
) = atoms

# MDMC does not add improper dihedrals interactions for us, we need to
# add them in manually
impropers = [
    DihedralAngle((C807, O808, C809, C806), improper=True),
    DihedralAngle((C804, C810, N803, C805), improper=True),
    DihedralAngle((N803, C801, C804, H814), improper=True),
    DihedralAngle((C805, C804, C806, H815), improper=True),
    DihedralAngle((C806, H816, C805, C807), improper=True),
    DihedralAngle((C809, H818, C810, C807), improper=True),
    DihedralAngle((C810, C809, H819, C804), improper=True),
    DihedralAngle((C801, N803, C800, O802), improper=True),
]


# Create a universe and add the paracetamol molecules
universe = Universe(dimensions=15.0)
universe.fill(paracetamol, num_density=0.005)
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

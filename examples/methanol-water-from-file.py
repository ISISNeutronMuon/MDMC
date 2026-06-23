import os

from MDMC.MD import *
from MDMC.MD.force_fields.OPLSAA import add_opls_force_field
from MDMC.exporters.trajectories.H5MD_build import write_H5MD

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)

atom_type_mapping = {
"H1" : {"name":"98", "atom_type":"98"},
"H2" : {"name":"98", "atom_type":"98"},
"H3" : {"name":"98", "atom_type":"98"},
"C" : {"name":"99", "atom_type":"99"},
"O" : {"name":"96", "atom_type":"96"},
"H4" : {"name":"97", "atom_type":"97"},

# the water atom names match those in the PDB file.
"OW" : {"name":"63", "atom_type":"63"},
"HW1" : {"name":"64", "atom_type":"64"},
"HW2" : {"name":"64", "atom_type":"64"},
}

bonds = {
    "MET": [("C", "H1"), ("C", "H2"), ("C", "H3"), ("O", "H4"), ("C", "O"), ],
    "SOL": [("HW1", "OW"), ("HW2", "OW")]
}

angles = {
    "MET": [("H1", "C", "O"), ("H2", "C", "O"),("H3", "C", "O"),
            ("H1", "C", "H2"), ("H2", "C", "H3"), ("H3", "C", "H1"),
            ("H4", "O", "C"),],
    "SOL": [("HW1", "OW", "HW2")]
}

dihedrals = {
    "MET": [("H1", "C", "O", "H4"), ("H2", "C", "O", "H4"), ("H3", "C", "O", "H4")],
    "SOL": []
}


# Create the universe from file
universe = Universe.from_pdb_file("structure/METHANOL_WATER.pdb",
                                  atom_type_mapping=atom_type_mapping,
                                  bonds_per_molecule=bonds,
                                  angles_per_molecule=angles,
                                  dihedrals_per_molecule=dihedrals)
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

simulation.run(n_steps=3000)

write_H5MD(simulation.trajectory,
           "one_run_Me_H2O_from_file.h5",
           timestamp="",
           creator_name="MDMC",
           creator_email="place@hold.er")

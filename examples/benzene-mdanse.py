import copy
import warnings

from statsmodels.tools.sm_exceptions import InterpolationWarning
from openmm import unit

from MDMC.readers.configurations import read
from MDMC.control import Control
from MDMC.MD import *
from MDMC.readers.observables.xml_SQw import XML_SQw
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.mdanse_observable import (
    MDANSEObservable,
    create_mdanse_resolution,
    MDANSE_RESOLUTION_FUNCTIONS,
)
from MDMC.MD.force_fields.OPLSAA import add_opls_force_field


warnings.filterwarnings("ignore", category=InterpolationWarning)


def run_everything():
    benzene_atoms = read("benzene.pdb", name=[
        # numbers below tell MDMC which OPLS parameters to use see
        # MDMC/MD/force_fields/data/oplsaa.dat
        "90", "90", "90", "90", "90", "90",  # Aromatic C
        "91", "91", "91", "91", "91", "91"  # Aromatic H-C H
    ])
    benzene = Molecule(atoms=benzene_atoms, name="benzene")

    # set the same labels so it's the same as libpargen (using benzene.pdb
    # as the input https://jorgensenresearch.com/ligpargen) for easy comparison
    # using the above pdb as input and openmm xml as output
    (
        C804, C805, C801, C800, C802, C803,  # Aromatic C
        H806, H807, H808, H809, H810, H811  # Aromatic H-C H
    ) = benzene_atoms

    # MDMC did not add the bond angles, and proper and improper dihedrals
    # we have to add them in manually
    BondAngle((C802, C800, C801))
    BondAngle((C803, C802, C800))
    BondAngle((C804, C803, C802))
    BondAngle((C805, C804, C803))
    BondAngle((H806, C804, C803))
    BondAngle((H807, C805, C804))
    BondAngle((H808, C801, C800))
    BondAngle((H809, C800, C801))
    BondAngle((H810, C802, C800))
    BondAngle((H811, C803, C802))
    BondAngle((H809, C800, C802))
    BondAngle((H810, C802, C803))
    BondAngle((H811, C803, C804))
    BondAngle((H806, C804, C805))
    BondAngle((C800, C801, C805))
    BondAngle((H808, C801, C805))
    BondAngle((C801, C805, H807))
    BondAngle((C804, C805, C801))
    DihedralAngle((C803, C802, C800, C801))
    DihedralAngle((C804, C803, C802, C800))
    DihedralAngle((C805, C804, C803, C802))
    DihedralAngle((H806, C804, C803, C802))
    DihedralAngle((H807, C805, C804, C803))
    DihedralAngle((H808, C801, C800, C802))
    DihedralAngle((H809, C800, C801, C805))
    DihedralAngle((H810, C802, C800, C801))
    DihedralAngle((H811, C803, C802, C800))
    DihedralAngle((C805, C801, C800, C802))
    DihedralAngle((H809, C800, C802, C803))
    DihedralAngle((H810, C802, C803, C804))
    DihedralAngle((H811, C803, C804, C805))
    DihedralAngle((H811, C803, C804, H806))
    DihedralAngle((C801, C805, C804, H806))
    DihedralAngle((H807, C805, C804, H806))
    DihedralAngle((C800, C801, C805, H807))
    DihedralAngle((H808, C801, C805, H807))
    DihedralAngle((H809, C800, C801, H808))
    DihedralAngle((C804, C805, C801, H808))
    DihedralAngle((H810, C802, C800, H809))
    DihedralAngle((H811, C803, C802, H810))
    DihedralAngle((C803, C804, C805, C801))
    DihedralAngle((C800, C801, C805, C804))
    DihedralAngle((C800, C801, C802, H809), improper=True)
    DihedralAngle((C801, C800, C805, H808), improper=True)
    DihedralAngle((C802, C800, C803, H810), improper=True)
    DihedralAngle((C803, C802, C804, H811), improper=True)
    DihedralAngle((C804, C803, C805, H806), improper=True)
    DihedralAngle((C805, C804, C801, H807), improper=True)

    # Create a universe and add the benzene molecules
    universe = Universe(dimensions=20.0)
    universe.fill(benzene, num_struc_units=55)
    add_opls_force_field(universe, cutoff=6.0, ewald=1e-4)  # needs adjusting

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
        openmm_nonbonded_combining="GEOMETRIC",
        openmm_ensembles=[
            # equilibration stage 1 equilibrate the cell volume and temperature
            # with high friction
            {
                "integrator": "LangevinMiddle",
                "frictionCoeff": 2.5 / unit.picoseconds,
                "barostat": {
                    "barostat": "MonteCarlo",
                    "defaultPressure": 1.01 * unit.bar,
                },
                # runs auto-equilibration using the KPSS test on certain properties
                # this tuple can be replaced with an int if you prefer to run
                # a specific number of steps instead.
                # auto-equilibration parameters are as follows:
                # (ensemble [NPT runs KPSS test on volume and temperature],
                # max number of steps, steps per iteration, kpss window,
                # kpss tolerance)
                "n_steps": ("NPT", 100000, 100, 1000, 0.01) # probably need to change all these parameters
            },
            # equilibration stage 2 equilibrate the cell volume and temperature
            # with more normal friction and monte carlo pressure changes
            {
                "integrator": "LangevinMiddle",
                "frictionCoeff": 0.25 / unit.picoseconds,
                "barostat": {
                    "barostat": "MonteCarlo",
                    "defaultPressure": 20.1 * unit.bar,
                },
                "n_steps": ("NPT", 100000, 100, 1000, 0.01) # probably need to change all these parameters
            },
            # equilibration stage 3 NVT with the equilibrated cell volume
            {
                "integrator": "LangevinMiddle",
                "frictionCoeff": 0.25 / unit.picoseconds,
                "n_steps": ("NVT", 100000, 100, 1000, 0.01) # probably need to change all these parameters
            },
            # equilibration stage 4 NVE equilibration to prepare for production
            {
                "integrator": "Verlet",
                "n_steps": ("NVE", 100000, 100, 1000, 0.01) # probably need to change all these parameters
            },
            # production NVE, n_steps not specified here since this is
            # determined by MDMC using the expt data
            {
                "integrator": "Verlet",
            },
        ]
    )

    simulation.run(n_steps=30000, equilibration=True)

    # exp_datasets is a list of dictionaries with one dictionary per experimental
    # dataset
    exp_xml = "../doc/tutorials/data/Well_s_q_omega_Ar_data.xml"
    exp_datasets = [
        {
            "file_name": exp_xml,
            "type": "MDANSE",
            "reader": "xml_SQw",
            "weight": 1.0,
            "resolution": None,
            "cont_slicing": True,
        }
    ]

    print(f"Available resolution functions: {MDANSE_RESOLUTION_FUNCTIONS}")
    mdanse_resolution = create_mdanse_resolution(
        exp_datasets[0]["resolution"],
    )

    data_parser = XML_SQw(exp_xml)

    exp_observable = MDANSEObservable(mdanse_job_type="SQw")
    exp_observable.read_from_file(data_parser)
    md_observable = MDANSEObservable(mdanse_job_type="SQw")
    md_observable.origin = "MD"
    md_observable.independent_variables = copy.deepcopy(exp_observable.independent_variables)

    observable_pair = ObservablePair(
        exp_obs=exp_observable,
        MD_obs=md_observable,
        weight=1.0,
        rescale_factor=1.0,
        auto_scale=True,
    )

    # only optimise the lennard-jones parameters
    for p in universe.parameters.as_array:
        if p.parameter_name == "OPLS-90-nonbonded_epsilon":
            # OPLS values 0.29288 kJ / mol
            p.constraints = [0.25, 0.35]  # change to desired search space
        elif p.parameter_name == "OPLS-90-nonbonded_sigma":
            # OPLS values 3.55 Ang
            p.constraints = [3.0, 4.0]  # change to desired search space
        elif p.parameter_name == "OPLS-91-nonbonded_epsilon":
            # OPLS values 0.12552 kJ / mol
            p.constraints = [0.10, 0.15]  # change to desired search space
        elif p.parameter_name == "OPLS-91-nonbonded_sigma":
            # OPLS values 2.42 Ang
            p.constraints = [2.0, 3.0]  # change to desired search space
        else:
            p.fixed = True

    new_settings = {
        "running_mode": ("multicore", 8),
        "instrument_resolution": mdanse_resolution,
    }
    md_observable.set_parameters(new_settings)

    # Specify how the refinement is going to be controlled
    control = Control(
        simulation=simulation,
        exp_datasets=exp_datasets,
        fit_parameters=universe.parameters,
        observable_pairs=[observable_pair],
        equilibration_steps=9000,  # needs adjusting for system size and experimental data
        MD_steps=4800,  # needs adjusting for system size and experimental data
        cont_slicing=True,
        file_dump_extent="all",
        file_dump_frequency="best",
        FoM_options={"error": "none"},
        conv_tol=1e-6,
    )
    # Energy Minimization and equilibration
    control.equilibrate(n_steps=15000)

    # Run the refinement, i.e. refine the FF parameters against the data.
    control.refine(n_steps=200)


if __name__ == "__main__":
    run_everything()

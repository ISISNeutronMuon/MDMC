from unittest.mock import patch
import os

from MDMC.control import Control
from MDMC.MD import Atom, LennardJones, Simulation, Universe, InteractionFunction
from MDMC.MD.interactions import Dispersion

import numpy as np

from pathlib import Path

from asv_runner.benchmarks.mark import skip_for_params

class MockEngine:
    """A mock MD engine."""
    def run(self):
        pass

    def clear(self):
        pass

    def setup_universe(self, *args, **kwargs):
        pass

    def setup_simulation(self, *args, **kwargs):
        pass

    def convert_trajectory(self, *args, **kwargs):
        pass

    def update_parameters(self, *args, **kwargs):
        pass

    def save_config(self, *args, **kwargs):
        pass

    def reset_config(self, *args, **kwargs):
        pass

class MockSimulation(Simulation):
    """
    Mock the ``Simulation`` so that we do not setup the MD engine so we can run
    the tests without having an MD engine installed.
    """

    def __init__(self, universe: Universe, traj_step: int,
                 time_step: float = 1., **settings):
        self.universe = universe
        self.settings = settings
        self.engine = MockEngine()
        self.traj_step = traj_step
        self.time_step = time_step
        self.ran = False
        self.auto_equilibrated = False

    def run(self, *args, **kwargs):
        self.ran = True
        self.engine.run()

    def auto_equilibrate(self, *args, **kwargs):
        self.auto_equilibrated = True
        self.engine.run()


class MinimizerSuite:
    timeout = 10000

    def setup(self):
        density = 0.0176
    
        self.universe = Universe(dimensions=38.4441)
        
        Ar = Atom('Ar', charge=0.)

        n_ar_atoms = int(density * np.prod(self.universe.dimensions))

        self.universe.fill(Ar, num_struc_units=(n_ar_atoms))

        self.simulation = Simulation(self.universe,
                        engine="lammps",
                        time_step=10.18893,
                        temperature=120.,
                        traj_step=15)
        
        Ar_dispersion = Dispersion(self.universe,
                           (Ar.atom_type, Ar.atom_type),
                           cutoff=8.,
                           vdw_tail_correction=True,
                           function=LennardJones(1.0243, 3.36))


        self.simulation = Simulation(self.universe,
                        engine="lammps",
                        time_step=10.18893,
                        temperature=120.,
                        traj_step=15)
        
        data_path = Path(__file__).with_name("data")
        input_file_path = data_path.joinpath("Well_s_q_omega_Ar_data.xml")
        
        self.exp_datasets = [
            {
                'file_name':input_file_path.absolute(),
                'type':'SQw',
                'reader':'xml_SQw',
                'weight':1.,
                'resolution':None
            }
            ]
    
        self.control_MMC = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="MMC",
            n_steps=100
        )
        
        self.control_GPR = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="GPR",
            n_steps=100
        )
        
        self.control_GPO = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="GPO",
            n_steps=100
        )
    
    # def time_minimiser_MMC(self):
    #    fom = self.control_MMC.max_FoM
    #    self.control_MMC.minimizer.step(fom)

    # def time_minimiser_GPR(self):
    #    fom = self.control_GPR.max_FoM
    #    self.control_GPR.minimizer.step(fom)

    # def time_minimiser_GPO(self):
    #    fom = self.control_GPO.max_FoM
    #    self.control_GPO.minimizer.step(fom)


class RefineSuite:
    timeout = 10000
    n_params = [1, 3, 5, 10]
    n_steps = [5, 10, 50, 100]

    params = (n_params, n_steps)
    param_names = ["Number of parameters", "Number of steps"]

    def setup(self, n_params, n_steps):
        os.environ["OMP_NUM_THREADS"] = "8"
        density = 0.0176
    
        self.universe = Universe(dimensions=10.)
        
        Ar = Atom('Ar', charge=0.)

        n_ar_atoms = int(density * np.prod(self.universe.dimensions))

        self.universe.fill(Ar, num_struc_units=(n_ar_atoms))

        self.simulation = MockSimulation(self.universe,
                        time_step=10.188949,
                        temperature=120.,
                        traj_step=15)
        
        params = {str(i): np.random.uniform(-1., 1.) for i in range(n_params)}

        interactions = InteractionFunction(params)

        for p in interactions.parameters.values():
            p.constraints = (-1., 1.)
        
        Ar_dispersion = Dispersion(self.universe,
                           (Ar.atom_type, Ar.atom_type),
                           cutoff=8.,
                           vdw_tail_correction=True,
                           function=interactions)

        data_path = Path(__file__).with_name("data")
        input_file_path = data_path.joinpath("Well_s_q_omega_Ar_data.xml")
        
        self.exp_datasets = [
            {
                'file_name':input_file_path.absolute(),
                'type':'SQw',
                'reader':'xml_SQw',
                'weight':1.,
                'resolution':None
            }
            ]
    
        self.control_MMC = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="MMC",
            n_steps=n_steps
        )
        
        self.control_GPR = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="GPR",
            n_steps=n_steps
        )
        
        self.control_GPO = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="GPO",
            n_steps=n_steps
        )

    def mock_FoM(self):
        self.fom = 0

        for v in self.fit_parameters.values():
            self.fom += v.value ** 2

        return self.fom

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_refineMMC(self, n_params, n_steps):
        self.control_MMC.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_refineMMC(self, n_params, n_steps):
        self.control_MMC.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_refineMMC(self, n_params, n_steps):
        self.control_MMC.refine(n_steps=n_steps)
        return self.control_MMC.fom

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_refineGPO(self, n_params, n_steps):
        self.control_GPO.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_refineGPO(self, n_params, n_steps):
        self.control_GPO.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_refineGPO(self, n_params, n_steps):
        self.control_GPO.refine(n_steps=n_steps)
        return float(self.control_GPO.fom)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_refineGPR(self, num_params, n_steps):
        self.control_GPR.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_refineGPR(self, n_params, n_steps):
        self.control_GPR.refine(n_steps=n_steps)
        return float(self.control_GPR.fom)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_refineGPR(self, num_params, n_steps):
        self.control_GPR.refine(n_steps=n_steps)

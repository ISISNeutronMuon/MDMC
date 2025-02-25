"""Benchmarking suites, to be run using asv during CI"""
from unittest.mock import patch
import os
from pathlib import Path
from itertools import product

import numpy as np

from MDMC.control import Control
from MDMC.MD import Atom, Simulation, Universe, InteractionFunction
from MDMC.MD.interactions import Dispersion

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
    the benchmarks without running MD.
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
        
def setup_controls(n_params, n_steps):
    """
    Setup method, to be called by asv before each benchmark is run.
    Sets up control objects for each minimizer type.

    Parameters
    ----------
    n_params: int
        Number of parameters to use in the benchmark

    n_steps : int
        Number of steps to run the refinement for in the benchmark
        
    Returns
        -------
        control_MMC : MDMC.control.Control
            A control object with MD mocked out and an MMC minimizer
            
        control_GPR : MDMC.control.Control
            A control object with MD mocked out and an GPR minimizer
            
        control_GPO : MDMC.control.Control
            A control object with MD mocked out and an GPO minimizer
    Examples
    --------
    Setup a benchmark with 5 parameters and 100 steps:

        .. highlight:: python
        .. code-block:: python

        minimizer_suite.setup(5, 100)
    """

    os.environ["OMP_NUM_THREADS"] = "8"
    density = 0.0176

    universe = Universe(dimensions=10.)

    Ar = Atom('Ar', charge=0.)

    n_ar_atoms = int(density * np.prod(universe.dimensions))

    universe.fill(Ar, num_struc_units=(n_ar_atoms))

    simulation = MockSimulation(universe,
                    time_step=10.188949,
                    temperature=120.,
                    traj_step=15)

    params = {str(i): np.random.uniform(400., 450.) for i in range(n_params)}

    interactions = InteractionFunction(params)

    for p in interactions.parameters.values():
        p.constraints = (400., 450.)

    _ = Dispersion(universe,
                        (Ar.atom_type, Ar.atom_type),
                        cutoff=8.,
                        vdw_tail_correction=True,
                        function=interactions)

    data_path = Path(__file__).with_name("data")
    input_file_path = data_path.joinpath("Well_s_q_omega_Ar_data.xml")

    exp_datasets = [
        {
            'file_name':input_file_path.absolute(),
            'type':'SQw',
            'reader':'xml_SQw',
            'weight':1.,
            'resolution':None
        }
    ]

    control_MMC = Control(
        simulation=simulation,
        exp_datasets=exp_datasets,
        fit_parameters=universe.parameters,
        MD_steps=570,
        minimizer_type="MMC",
        n_steps=n_steps
    )

    control_GPR = Control(
        simulation=simulation,
        exp_datasets=exp_datasets,
        fit_parameters=universe.parameters,
        MD_steps=570,
        minimizer_type="GPR",
        n_steps=n_steps
    )

    control_GPO = Control(
        simulation=simulation,
        exp_datasets=exp_datasets,
        fit_parameters=universe.parameters,
        MD_steps=570,
        minimizer_type="GPO",
        n_steps=n_steps
    )
    
    return control_MMC, control_GPR, control_GPO

def mock_FoM(self):
    """
    Mock figure of merit method.
    Calculates the Schwefel function of the parameters in the control
    object it is mocking.
    """
    vals = np.array([v.value for v in self.fit_parameters.values()])
    
    self.fom = schwefel(vals)
    
    return self.fom


def schwefel(x):
    """
    Calculates the Scwefel function (https://www.sfu.ca/~ssurjano/schwef.html) 
    of a given set of parameters.
    Provides a relatively complex function to optimise whilst still having a known global minimum.
    """
    d = x.size

    return (418.9829 * d) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    

class MinimizerSuite:
    """
    Suite of asv (https://asv.readthedocs.io/en/latest/) benchmarks
    Tests the performance of the different minimizers available:
    MMC, GPO and GPR.
    Uses a mocked FoM calculation so time spent on MD is not included.
    """

    timeout = 10000
    n_params = [1, 3, 5, 10]
    n_steps = [5, 10, 50]

    params = (n_params, n_steps)
    param_names = ["Number of parameters", "Number of steps"]

    def setup(self, n_params, n_steps):
        self.control_MMC, self.control_GPR, self.control_GPO = setup_controls(n_params, n_steps)

    #Although the benchmarks don't use n_params, asv will still pass it
    #so it needs to be included as an argument
    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_MMC(self, n_params, n_steps):
        """
        Time MMC minimizer
        """
        self.control_MMC.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_MMC(self, n_params, n_steps):
        """
        Record peak memory for MMC minimizer
        """
        self.control_MMC.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_MMC(self, n_params, n_steps):
        """
        Record FoM calculated by MMC minimizer
        """
        self.control_MMC.refine(n_steps=n_steps)
        return self.control_MMC.fom

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_GPO(self, n_params, n_steps):
        """
        Time GPO minimizer
        """
        self.control_GPO.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_GPO(self, n_params, n_steps):
        """
        Record peak memory for GPO minimizer
        """
        self.control_GPO.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_GPO(self, n_params, n_steps):
        """
        Record FoM calculated by GPO minimizer
        """
        self.control_GPO.refine(n_steps=n_steps)
        return float(self.control_GPO.fom)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_GPR(self, num_params, n_steps):
        """
        Time GPR minimizer
        """
        self.control_GPR.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_GPR(self, num_params, n_steps):
        """
        Record peak memory for GPR minimizer
        """
        self.control_GPR.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_GPR(self, n_params, n_steps):
        """
        Record FoM calculated by GPR minimizer
        """
        self.control_GPR.refine(n_steps=n_steps)
        return float(self.control_GPR.fom)

class MinimizerSuiteLong:
    """
    Suite of asv (https://asv.readthedocs.io/en/latest/) benchmarks
    Tests the performance of the different minimizers available:
    MMC, GPO and GPR.
    Uses a mocked FoM calculation so time spent on MD is not included.
    Runs benchmarks with more steps that would take too long to run for every PR
    """

    timeout = 10000
    n_params = [1, 3, 5, 10]
    n_steps = [100]

    params = (n_params, n_steps)
    param_names = ["Number of parameters", "Number of steps"]

    def setup(self, n_params, n_steps):
        self.control_MMC, self.control_GPR, self.control_GPO = setup_controls(n_params, n_steps)

    #Although the benchmarks don't use n_params, asv will still pass it
    #so it needs to be included as an argument
    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_MMC_long(self, n_params, n_steps):
        """
        Time MMC minimizer
        """
        self.control_MMC.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_MMC_long(self, n_params, n_steps):
        """
        Record peak memory for MMC minimizer
        """
        self.control_MMC.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_MMC_long(self, n_params, n_steps):
        """
        Record FoM calculated by MMC minimizer
        """
        self.control_MMC.refine(n_steps=n_steps)
        return self.control_MMC.fom

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_GPO_long(self, n_params, n_steps):
        """
        Time GPO minimizer
        """
        self.control_GPO.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_GPO_long(self, n_params, n_steps):
        """
        Record peak memory for GPO minimizer
        """
        self.control_GPO.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_GPO_long(self, n_params, n_steps):
        """
        Record FoM calculated by GPO minimizer
        """
        self.control_GPO.refine(n_steps=n_steps)
        return float(self.control_GPO.fom)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def time_GPR_long(self, num_params, n_steps):
        """
        Time GPR minimizer
        """
        self.control_GPR.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def peakmem_GPR_long(self, num_params, n_steps):
        """
        Record peak memory for GPR minimizer
        """
        self.control_GPR.refine(n_steps=n_steps)

    @patch.object(Control, "_generate_FoM", mock_FoM)
    def track_GPR_long(self, n_params, n_steps):
        """
        Record FoM calculated by GPR minimizer
        """
        self.control_GPR.refine(n_steps=n_steps)
        return float(self.control_GPR.fom)
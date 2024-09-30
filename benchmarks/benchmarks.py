from MDMC.control import Control
from MDMC.MD import Atom, LennardJones, Simulation, Universe
from MDMC.MD.interactions import Dispersion

import numpy as np

from pathlib import Path

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
            minimizer_type="MMC")
        
        self.control_GPR = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="GPR",
            n_steps=100)
        
        self.control_GPO = Control(
            simulation=self.simulation,
            exp_datasets=self.exp_datasets,
            fit_parameters=self.universe.parameters,
            MD_steps=570,
            minimizer_type="GPO",
            n_steps=100)
        
    def time_minimiser_MMC(self):
       fom = self.control_MMC.max_FoM
       self.control_MMC.minimizer.step(fom)

    def time_minimiser_GPR(self):
       fom = self.control_GPR.max_FoM
       self.control_GPR.minimizer.step(fom)

    def time_minimiser_GPO(self):
       fom = self.control_GPO.max_FoM
       self.control_GPO.minimizer.step(fom)


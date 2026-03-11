import logging
from typing import Any

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import Topology, Simulation

from MDMC.MD import LennardJones
from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory


LOGGER = logging.getLogger(__name__)


class OpenMMEngine(MDEngine):

    def __init__(self):
        super().__init__()
        self.universe = None
        self.openmm_system = None
        self.openmm_simulation = None
        self.openmm_platform = mm.Platform.getPlatformByName("CPU")

    @property
    def saved_config(self) -> 'Configuration':
        raise NotImplementedError

    def setup_universe(self, universe: str, **settings: dict) -> None:
        self.universe = universe
        self.openmm_system = mm.System()
        box_matrix = np.array([
            [self.universe.dimensions[0], 0, 0],
            [0, self.universe.dimensions[1], 0],
            [0, 0, self.universe.dimensions[2]]
        ]) * unit.nanometer

        self.openmm_system.setDefaultPeriodicBoxVectors(*box_matrix)

        for type_ID, atom_type_group in universe.atom_types.items():
            for _ in atom_type_group:
                self.openmm_system.addParticle(float(atom_type_group[0].mass) * unit.amu)

        for disp in self.universe.nonbonded_interactions:
            if not isinstance(disp.function, LennardJones):
                LOGGER.warning("Only LennardJones potential currently supported.")
                continue
            sigma = float(disp.function.sigma.value) * unit.angstrom
            epsilon = float(disp.function.epsilon.value) * unit.kilojoules_per_mole
            force = mm.NonbondedForce()
            force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
            for type_ID, atom_type_group in universe.atom_types.items():
                for _ in atom_type_group:
                    force.addParticle(0.0, sigma, epsilon)
            force.setCutoffDistance(disp.cutoff)
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(0.8 * disp.cutoff)
            force.setUseDispersionCorrection(True)
            self.openmm_system.addForce(force)

    def setup_simulation(self, **settings: dict) -> None:
        temperature = settings.get("temperature")

        compound_integrator = mm.CompoundIntegrator()
        lang_int_1 = mm.LangevinMiddleIntegrator(
            temperature,
            10.0 / unit.picoseconds,
            self.time_step * unit.femtoseconds
        )
        compound_integrator.addIntegrator(lang_int_1)
        lang_int_2 = mm.LangevinMiddleIntegrator(
            temperature,
            1.0 / unit.picoseconds,
            self.time_step * unit.femtoseconds
        )
        compound_integrator.addIntegrator(lang_int_2)
        compound_integrator.addIntegrator(mm.VerletIntegrator(self.time_step * unit.femtoseconds))

        self.openmm_simulation = Simulation(
            Topology(),
            self.openmm_system,
            compound_integrator,
            self.openmm_platform
        )

        positions = np.array([atom.position for atom in self.universe.atoms]) * unit.angstrom
        self.openmm_simulation.context.setPositions(positions)
        self.openmm_simulation.context.setVelocitiesToTemperature(temperature)

    def minimize(self, n_steps: int, minimize_every: int = 10,
                 **settings: dict) -> None:
        self.openmm_simulation.minimizeEnergy(maxIterations=n_steps)

    def run(self, n_steps: int, equilibration: bool, output_log: str = None,
            work_dir: str = None, **settings: dict) -> None:
        if equilibration:
            self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(0)
            self.openmm_simulation.step(n_steps // 2)
            self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(1)
            self.openmm_simulation.step(n_steps // 2)
        else:
            self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(2)
            self.openmm_simulation.step(n_steps)

    def convert_trajectory(self, start: int = 0, stop: int = None,
                           step: int = 1, **settings: dict) -> 'CompactTrajectory':
        traj = CompactTrajectory()

        raise NotImplementedError

    def update_parameters(self) -> None:
        raise NotImplementedError

    def save_config(self) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def reset_config(self) -> None:
        raise NotImplementedError

    def eval(self, variable: str) -> Any:
        raise NotImplementedError
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
        self.compact_trajectory = None

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
        ]) * unit.angstrom

        self.openmm_system.setDefaultPeriodicBoxVectors(*box_matrix)

        for type_ID, atom_type_group in universe.atom_types.items():
            for _ in atom_type_group:
                self.openmm_system.addParticle(float(atom_type_group[0].mass) * unit.amu)

        for disp in universe.nonbonded_interactions:
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
            force.setCutoffDistance(disp.cutoff * unit.angstrom)
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(0.8 * disp.cutoff * unit.angstrom)
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
            self.openmm_platform,
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
            self.compact_trajectory = CompactTrajectory()
            self.compact_trajectory.preAllocate(n_steps=n_steps, n_atoms=len(self.universe.atoms))
            self.compact_trajectory.validateTypes([atom.atom_type for atom in self.universe.atoms])
            reporter = CompactTrajectoryReporter(self.compact_trajectory, self.traj_step, n_steps)
            self.openmm_simulation.reporters.append(reporter)
            self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(2)
            self.openmm_simulation.currentStep = 0
            self.openmm_simulation.context.setTime(0.0)
            state = self.openmm_simulation.context.getState(
                getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
            )
            reporter.report(self.openmm_simulation, state)
            self.openmm_simulation.step(n_steps)
            self.openmm_simulation.reporters.clear()

    def convert_trajectory(self, start: int = 0, stop: int = None,
                           step: int = 1, **settings: dict) -> CompactTrajectory:
        traj = self.compact_trajectory
        atom_symbols = [atom.element.symbol for atom in self.universe.atoms]
        atom_masses = [atom.mass for atom in self.universe.atoms]
        traj.labelAtoms(atom_symbols, atom_masses)
        traj.setCharge([atom.charge for atom in self.universe.atoms])
        traj.postProcess()
        self.compact_trajectory = None
        return traj

    def update_parameters(self) -> None:
        i = 0
        for disp in self.universe.nonbonded_interactions:
            if not isinstance(disp.function, LennardJones):
                continue
            sigma = float(disp.function.sigma.value) * unit.angstrom
            epsilon = float(disp.function.epsilon.value) * unit.kilojoules_per_mole

            force = self.openmm_simulation.system.getForce(i)

            j = 0
            for type_ID, atom_type_group in  self.universe.atom_types.items():
                for _ in atom_type_group:
                    force.setParticleParameters(j, 0.0, sigma, epsilon)
                    j += 1

            force.updateParametersInContext(self.openmm_simulation.context)
            i += 1

    def save_config(self) -> None:
        pass

    def clear(self) -> None:
        raise NotImplementedError

    def reset_config(self) -> None:
        raise NotImplementedError

    def eval(self, variable: str) -> Any:
        raise NotImplementedError


class CompactTrajectoryReporter:
    def __init__(self, compact_trajectory, report_interval, n_steps):
        self.compact_trajectory = compact_trajectory
        self.report_interval = report_interval
        self.n_steps = n_steps

    def report(self, simulation, state):
        step = simulation.currentStep

        if step >= self.n_steps:
            return

        time = state.getTime().value_in_unit(unit.femtoseconds)
        positions = state.getPositions().value_in_unit(unit.angstrom)

        # currently MDMC can only deal with cubic lattice
        a, b, c = state.getPeriodicBoxVectors()
        a = np.array(a.value_in_unit(unit.angstrom))[0]
        b = np.array(b.value_in_unit(unit.angstrom))[1]
        c = np.array(c.value_in_unit(unit.angstrom))[2]

        self.compact_trajectory.writeOneStep(
            step_num=step // self.report_interval,
            time=time,
            positions=np.array(positions),
        )
        self.compact_trajectory.setDimensions(np.array([a, b, c]), step_num=step)

    def describeNextReport(self, simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval

        if simulation.currentStep + steps >= self.n_steps:
            return steps, False, False, False, False
            
        return steps, True, True, True, True

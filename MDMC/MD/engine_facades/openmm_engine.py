import itertools as it
import logging
from typing import Any

import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import Simulation, Topology

from MDMC.MD import NonBonded
from MDMC.MD.interactions import NonBondedForce, HarmonicPotentialForce
from MDMC.MD.engine_facades.facade import MDEngine, MDEngineError
from MDMC.MD.simulation import Universe
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

LOGGER = logging.getLogger(__name__)


class OpenMMEngine(MDEngine):
    def __init__(self):
        super().__init__()
        self.universe = None
        self.openmm_system = None
        self.openmm_simulation = None
        self.compact_trajectory = None
        self.temperature = None
        self._saved_config = None

    @property
    def saved_config(self) -> np.ndarray:
        """Get the saved configuration of the atomic positions.

        Returns
        -------
        np.ndarray
            The atomic positions.
        """
        if self._saved_config is None:
            raise TypeError("OpenMMEngine has not been run.")
        return self._saved_config

    def setup_universe(self, universe: Universe, **settings: Any) -> None:
        """Set up the openmm system.

        Parameters
        ----------
        universe : Universe
            A molecular dynamics ``Universe`` which will be setup in the
            ``OpenMMEngine``.
        **settings
            Not used.
        """
        self.universe = universe
        self.openmm_system = mm.System()
        # currently MDMC can only deal with orthorhombic lattices
        box_matrix = (
            np.array(
                [
                    [self.universe.dimensions[0], 0, 0],
                    [0, self.universe.dimensions[1], 0],
                    [0, 0, self.universe.dimensions[2]],
                ],
            )
            * unit.angstrom
        )

        self.openmm_system.setDefaultPeriodicBoxVectors(*box_matrix)

        for atom in self.universe.atoms:
            self.openmm_system.addParticle(atom.mass * unit.amu)

        self.change_openmm_force_field()

    def change_openmm_force_field(self):
        """Change the OpenMM force fields."""
        openmm_nobonded = mm.NonbondedForce()
        for atom in self.universe.atoms:
            nonbonded = [
                force.function
                for force in atom.nonbonded_interactions
                if isinstance(force.function, NonBonded)
            ]
            if len(nonbonded) != 1:
                raise Exception("Unexpected number of non-bonded interactions.")
            nonbonded = nonbonded[0]
            charge = nonbonded.charge.value * unit.coulomb
            sigma = nonbonded.sigma.value * unit.angstrom
            epsilon = nonbonded.epsilon.value * unit.kilojoules_per_mole
            openmm_nobonded.addParticle(charge, sigma, epsilon)

        mdmc_nonbonded = [
            force for force in self.universe.interactions if isinstance(force, NonBondedForce)
        ]
        cutoff = max(force.cutoff for force in mdmc_nonbonded)
        ewald = min(force.ewald for force in mdmc_nonbonded)
        openmm_nobonded.setNonbondedMethod(mm.NonbondedForce.PME)
        openmm_nobonded.setCutoffDistance(cutoff * unit.angstrom)
        openmm_nobonded.setEwaldErrorTolerance(ewald)
        openmm_nobonded.setUseSwitchingFunction(False)
        openmm_nobonded.setUseDispersionCorrection(False)
        self.openmm_system.addForce(openmm_nobonded)

        bond_force = mm.HarmonicBondForce()
        mdmc_harmonic = [
            force
            for force in self.universe.interactions
            if isinstance(force, HarmonicPotentialForce)
        ]
        for force in mdmc_harmonic:
            equil_length = force.function.equilibrium_state.value * unit.angstrom
            force_const = (
                force.function.potential_strength.value
                * unit.kilojoules_per_mole
                / unit.angstrom**2
            )
            for atm_i, atm_j in it.combinations(force.atoms, 2):
                if (atm_i, atm_j) not in force.atom_types:
                    continue
                bond_force.addBond(atm_i.ID - 1, atm_j.ID - 1, equil_length, force_const)
        self.openmm_system.addForce(bond_force)

    def clear_forces(self):
        """Clear the OpenMM force fields from the OpenMM system."""
        for i in reversed(range(self.openmm_system.getNumForces())):
            self.openmm_system.removeForce(i)

    def setup_simulation(self, **settings: Any) -> None:
        """Set up the openmm simulation.

        Parameters
        ----------
        settings**
            Some settings which are used to set up the openmm
            simulation object.
        """
        self.temperature = settings.get("temperature")

        compound_integrator = mm.CompoundIntegrator()
        lang_int_1 = mm.LangevinMiddleIntegrator(
            self.temperature,
            10.0 / unit.picoseconds,
            self.time_step * unit.femtoseconds,
        )
        compound_integrator.addIntegrator(lang_int_1)
        lang_int_2 = mm.LangevinMiddleIntegrator(
            self.temperature,
            1.0 / unit.picoseconds,
            self.time_step * unit.femtoseconds,
        )
        compound_integrator.addIntegrator(lang_int_2)
        compound_integrator.addIntegrator(mm.VerletIntegrator(self.time_step * unit.femtoseconds))

        self.openmm_simulation = Simulation(
            Topology(),
            self.openmm_system,
            compound_integrator,
            mm.Platform.getPlatformByName(settings.get("openmm_platform")),
        )

        positions = np.array([atom.position for atom in self.universe.atoms]) * unit.angstrom
        self.openmm_simulation.context.setPositions(positions)
        self.openmm_simulation.context.setVelocitiesToTemperature(self.temperature)

    def minimize(self, n_steps: int, minimize_every: int = 10, **settings: Any) -> None:
        """Minimizes the simulation energy.

        Parameters
        ----------
        n_steps : int
            Maximum number of iterations during the energy minimization.
        minimize_every : int, optional, default 10
            Not used.
        """
        self.openmm_simulation.minimizeEnergy(maxIterations=n_steps)

    def run(
        self,
        n_steps: int,
        equilibration: bool,
        output_log: str = None,
        work_dir: str = None,
        **settings: Any,
    ) -> None:
        """Run the simulation.

        Parameters
        ----------
        n_steps : int
            Number of steps for the time integrator.
        equilibration : bool
            If `True`, run is equilibration which does not store the
            ``trajectory`` otherwise run is production.
        output_log: str, optional, default None
            Not used.
        work_dir: str, optional, default None
            Not used.
        **settings
            Not used.
        """
        if equilibration:
            try:
                self.openmm_simulation.minimizeEnergy()
                self.openmm_simulation.context.setVelocitiesToTemperature(self.temperature)
                self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(0)
                self.openmm_simulation.step(n_steps // 3)
                self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(1)
                self.openmm_simulation.step(n_steps // 3)
                self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(2)
                self.openmm_simulation.step(n_steps // 3)
            except mm.OpenMMException as e:
                LOGGER.warning(f"OpenMM exception during equilibration: {e}")
                raise MDEngineError(f"OpenMM exception during equilibration: {e}") from e
        else:
            self.compact_trajectory = CompactTrajectory()
            self.compact_trajectory.preAllocate(n_steps=n_steps, n_atoms=len(self.universe.atoms))
            reporter = CompactTrajectoryReporter(self.compact_trajectory, self.traj_step, n_steps)
            self.openmm_simulation.reporters.append(reporter)
            self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(2)
            self.openmm_simulation.currentStep = 0
            self.openmm_simulation.context.setTime(0.0)
            state = self.openmm_simulation.context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=True,
            )
            reporter.report(self.openmm_simulation, state)
            try:
                self.openmm_simulation.step(n_steps)
            except mm.OpenMMException as e:
                LOGGER.warning(f"OpenMM exception during production run: {e}")
                raise MDEngineError(f"OpenMM exception during production run: {e}") from e
            finally:
                self.openmm_simulation.reporters.clear()

    def convert_trajectory(
        self,
        start: int = 0,
        stop: int = None,
        step: int = 1,
        **settings: Any,
    ) -> CompactTrajectory:
        """Returns the MDMC compact trajectory.

        Parameters
        ----------
        start : int
            Not used.
        stop : int
            Not used.
        step : int
            Not used.
        **settings
            Not used.

        Returns
        -------
        CompactTrajectory
            The ``CompactTrajectory`` from the most recent production simulation.
        """
        self.compact_trajectory.validateTypes([atom.atom_type for atom in self.universe.atoms])
        atom_elements = [atom.element for atom in self.universe.atoms]
        atom_masses = [atom.mass for atom in self.universe.atoms]
        self.compact_trajectory.labelAtoms(atom_elements, atom_masses)
        self.compact_trajectory.setCharge([atom.charge for atom in self.universe.atoms])
        self.compact_trajectory.postProcess()
        return self.compact_trajectory

    def update_parameters(self) -> None:
        """Updates the ``OpenMMEngine`` force field parameters."""
        self.clear_forces()
        self.change_openmm_force_field()
        self.openmm_simulation.context.reinitialize(preserveState=True)

    def save_config(self) -> None:
        """Sets ``self._saved_config`` to the current set of positions."""
        state = self.openmm_simulation.context.getState(getPositions=True)
        self._saved_config = np.array(state.getPositions().value_in_unit(unit.angstrom))

    def clear(self) -> None:
        pass

    def reset_config(self) -> None:
        """Resets the atomic positions of the simulation to that in ``saved_config``."""
        self.openmm_simulation.context.setPositions(self.saved_config * unit.angstrom)
        self.openmm_simulation.context.setVelocitiesToTemperature(self.temperature)

    def eval(self, variable: str) -> Any:
        raise NotImplementedError


class CompactTrajectoryReporter:
    def __init__(self, compact_trajectory: CompactTrajectory, report_interval: int, n_steps: int):
        """Reporter which saves MD results into the MDMC compact
        trajectory.

        Parameters
        ----------
        compact_trajectory : CompactTrajectory
            The MDMC compact trajectory object.
        report_interval : int
            The interval which the MD results will be saved.
        n_steps : int
            The total number of step of the simulation.
        """
        self.compact_trajectory = compact_trajectory
        self.report_interval = report_interval
        self.n_steps = n_steps

    def report(self, simulation: Simulation, state: mm.State):
        """Save the simulation data into the MDMC compact trajectory.

        Parameters
        ----------
        simulation : Simulation
            The openmm simulation object.
        state : mm.State
            The openmm state object.
        """
        step = simulation.currentStep

        if step >= self.n_steps:
            return

        time = state.getTime().value_in_unit(unit.femtoseconds)
        positions = state.getPositions().value_in_unit(unit.angstrom)

        # currently MDMC can only deal with orthorhombic lattices
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

    def describeNextReport(self, simulation: Simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval

        if simulation.currentStep + steps >= self.n_steps:
            return steps, False, False, False, False

        return steps, True, True, True, True

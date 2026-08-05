# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Facade for OpenMM MD engine."""

import logging
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np
from statsmodels.tsa.stattools import kpss
import openmm as mm
from openmm import unit
from openmm.app import Simulation, Topology

from MDMC.MD import NonBonded
from MDMC.MD.engine_facades.facade import MDEngine, MDEngineError
from MDMC.MD.interaction_functions import DummyInteractionFunction
from MDMC.MD.interactions import Bond, BondAngle, DihedralAngle, NonBondedForce
from MDMC.MD.simulation import Universe
from MDMC.MD.structures import AverageSite3P
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory

LOGGER = logging.getLogger(__name__)
BOLTZ = unit.Quantity(1.3806503e-23, unit.joule / unit.kelvin)


class CombiningRules(Enum):
    LORENTZBERTHLOT = 0
    GEOMETRIC = 1


class OpenMMEngine(MDEngine):
    def __init__(self):
        super().__init__()
        self.universe = None
        self.openmm_system = None
        self.openmm_simulation = None
        self.compact_trajectory = None
        self.temperature = None
        self._saved_config = None
        self.MDMC_ID_to_idx = {}
        self.bond_graph = nx.Graph()
        self.nonbonded_scaling = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 1.0, 0.5],
        ]
        self.nonbonded_combining = CombiningRules.LORENTZBERTHLOT
        self.real_atom = []
        # the default openmm engine equilibration and production
        # integrators and options
        self.openmm_ensembles = [
            {
                "integrator": "LangevinMiddle",
                "frictionCoeff": 10.0 / unit.picoseconds,
            },
            {
                "integrator": "LangevinMiddle",
                "frictionCoeff": 1.0 / unit.picoseconds,
            },
            {
                "integrator": "Verlet",
            },
            {
                "integrator": "Verlet",
            },
        ]

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
            Some settings which are used to set up the openmm engine.
        """
        self.universe = universe
        self.openmm_system = mm.System()
        self.nonbonded_scaling = settings.get("openmm_nonbonded_scaling", self.nonbonded_scaling)
        combining_rule = settings.get("openmm_nonbonded_combining", "LORENTZBERTHLOT")
        if combining_rule.upper() not in CombiningRules.__members__:
            raise ValueError(
                f"Combining rule option {combining_rule} is not valid, "
                f"use one of the following: {list(CombiningRules.__members__)}"
            )
        self.nonbonded_combining = CombiningRules[combining_rule.upper()]

        # set the unit cell, note that currently MDMC can only deal with
        # orthorhombic lattices
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

        # add atoms
        for i, atom in enumerate(self.universe.atoms):
            self.MDMC_ID_to_idx[atom.ID] = i
            self.openmm_system.addParticle(atom.mass * unit.amu)
            self.bond_graph.add_node(i)

        for atom in self.universe.atoms:
            if isinstance(atom, AverageSite3P):
                i = self.MDMC_ID_to_idx[atom.ID]
                j = self.MDMC_ID_to_idx[atom.particles[0].ID]
                k = self.MDMC_ID_to_idx[atom.particles[1].ID]
                m = self.MDMC_ID_to_idx[atom.particles[2].ID]
                w_j = atom.weights[0]
                w_k = atom.weights[1]
                w_m = atom.weights[2]
                self.openmm_system.setVirtualSite(
                    i,
                    mm.ThreeParticleAverageSite(j, k, m, w_j, w_k, w_m),
                )
                self.real_atom.append(False)
            else:
                self.real_atom.append(True)

        # build bond graph
        mdmc_bonds = [force for force in set(self.universe.interactions) if isinstance(force, Bond)]
        for mdmc_bond in mdmc_bonds:
            for atm_i, atm_j in mdmc_bond.atoms:
                i = self.MDMC_ID_to_idx[atm_i.ID]
                j = self.MDMC_ID_to_idx[atm_j.ID]
                self.bond_graph.add_edge(i, j)

        # add force field
        self.change_openmm_force_field_and_constraints()

    def change_openmm_force_field_and_constraints(self):
        """Change the OpenMM force fields and constraints."""

        # add harmonic bond forces
        bond_dists = {}
        bond_force = mm.HarmonicBondForce()
        mdmc_bonds = [force for force in set(self.universe.interactions) if isinstance(force, Bond)]
        for mdmc_bond in mdmc_bonds:
            if isinstance(mdmc_bond.function, DummyInteractionFunction):
                continue
            equil_length = float(mdmc_bond.function.equilibrium_state.value)
            force_const = (
                float(mdmc_bond.function.potential_strength.value)
                * unit.kilojoules_per_mole
                / unit.angstroms**2
            )
            for atm_i, atm_j in mdmc_bond.atoms:
                i = self.MDMC_ID_to_idx[atm_i.ID]
                j = self.MDMC_ID_to_idx[atm_j.ID]
                if mdmc_bond.constrained:
                    self.openmm_system.addConstraint(i, j, equil_length * unit.angstroms)
                    bond_dists[(i, j)] = equil_length
                    bond_dists[(j, i)] = equil_length
                else:
                    # in openmm HarmonicBondForce is 1/2 K (x_0 - x)**2
                    # in lammps and therefore MDMC it is without the 1/2
                    bond_force.addBond(i, j, equil_length * unit.angstroms, 2 * force_const)
        self.openmm_system.addForce(bond_force)

        # add harmonic angle forces
        angle_force = mm.HarmonicAngleForce()
        mdmc_bondangles = [
            force for force in set(self.universe.interactions) if isinstance(force, BondAngle)
        ]
        for mdmc_bondangle in mdmc_bondangles:
            if isinstance(mdmc_bondangle.function, DummyInteractionFunction):
                continue
            equil_angle = float(mdmc_bondangle.function.equilibrium_state.value)
            force_const = (
                float(mdmc_bondangle.function.potential_strength.value)
                * unit.kilojoules_per_mole
                / unit.radians**2
            )
            for atm_i, atm_j, atm_k in mdmc_bondangle.atoms:
                i = self.MDMC_ID_to_idx[atm_i.ID]
                j = self.MDMC_ID_to_idx[atm_j.ID]
                k = self.MDMC_ID_to_idx[atm_k.ID]
                if mdmc_bondangle.constrained:
                    b = bond_dists[(i, j)]
                    c = bond_dists[(j, k)]
                    a = (
                        np.sqrt(b**2 + c**2 - 2 * b * c * np.cos(np.deg2rad(equil_angle)))
                        * unit.angstroms
                    )
                    # openmm doesn't have angle constraints, we expect
                    # bond constraints to be already set if angle
                    # constraints are set so we assume that bond constraints
                    # were already made above and can add a length constraint
                    # to fix the angle
                    self.openmm_system.addConstraint(i, k, a)
                else:
                    # in openmm HarmonicBondForce is 1/2 K (theta_0 - theta)**2
                    # in lammps and therefore MDMC it is without the 1/2
                    angle_force.addAngle(i, j, k, equil_angle * unit.degrees, 2 * force_const)
        self.openmm_system.addForce(angle_force)

        # add periodic torsion forces
        dihedral = mm.PeriodicTorsionForce()
        mdmc_dihedrals = [
            force for force in set(self.universe.interactions) if isinstance(force, DihedralAngle)
        ]
        for mdmc_dihedral in mdmc_dihedrals:
            func = mdmc_dihedral.function
            params = func.parameters
            n_funcs = len(params) // 3
            for i in range(1, n_funcs + 1):
                force_const = float(getattr(func, f"K{i}").value)
                n = int(getattr(func, f"n{i}").value)
                d = float(getattr(func, f"d{i}").value)
                if force_const == 0.0:
                    continue
                for atm_i, atm_j, atm_k, atm_l in mdmc_dihedral.atoms:
                    i = self.MDMC_ID_to_idx[atm_i.ID]
                    j = self.MDMC_ID_to_idx[atm_j.ID]
                    k = self.MDMC_ID_to_idx[atm_k.ID]
                    l = self.MDMC_ID_to_idx[atm_l.ID]  # noqa: E741
                    dihedral.addTorsion(
                        i, j, k, l, n, d * unit.degrees, force_const * unit.kilojoules_per_mole
                    )
        self.openmm_system.addForce(dihedral)

        # add nonbonded forces
        nonbonded = mm.NonbondedForce()
        use_ewald = False
        for atom in self.universe.atoms:
            mdmc_nonbonded = [
                force.function
                for force in atom.nonbonded_interactions
                if isinstance(force.function, NonBonded)
            ]
            if len(mdmc_nonbonded) != 1:
                raise Exception("Unexpected number of non-bonded interactions.")
            mdmc_nonbonded = mdmc_nonbonded[0]
            charge = float(mdmc_nonbonded.charge.value) * unit.elementary_charge
            sigma = float(mdmc_nonbonded.sigma.value) * unit.angstrom
            epsilon = float(mdmc_nonbonded.epsilon.value) * unit.kilojoules_per_mole
            nonbonded.addParticle(charge, sigma, epsilon)
            if mdmc_nonbonded.charge.value != 0.0:
                use_ewald = True

        mdmc_nonbonded = [
            force for force in set(self.universe.interactions) if isinstance(force, NonBondedForce)
        ]
        cutoff = max(force.cutoff for force in mdmc_nonbonded)
        ewald = min(force.ewald for force in mdmc_nonbonded)
        if use_ewald:
            nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
            nonbonded.setEwaldErrorTolerance(ewald)
        else:
            nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        nonbonded.setCutoffDistance(cutoff * unit.angstrom)
        nonbonded.setUseSwitchingFunction(False)
        nonbonded.setUseDispersionCorrection(False)
        self.openmm_system.addForce(nonbonded)
        self.apply_nonbonded_rules(nonbonded)

        # add centre of mass remover force
        self.openmm_system.addForce(mm.CMMotionRemover())

    def apply_nonbonded_rules(self, nonbonded: mm.NonbondedForce):
        """Change the combining rules and scale the sigma and epsilon
        parameters of the nonbonded force. Adds a custrom force to the
        openmm system if needed.

        Parameters
        ----------
        nonbonded : mm.NonbondedForce
            The openmm NonbondedForce object.

        """
        if self.nonbonded_combining == CombiningRules.GEOMETRIC:
            custom = mm.CustomNonbondedForce(
                "4*epsilon*((sigma/r)^12-(sigma/r)^6); "
                "sigma=sqrt(sigma1*sigma2); "
                "epsilon=sqrt(epsilon1*epsilon2)"
            )

            custom.addPerParticleParameter("sigma")
            custom.addPerParticleParameter("epsilon")
            custom.setCutoffDistance(nonbonded.getCutoffDistance())
            custom.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
            custom.setUseSwitchingFunction(False)
            self.openmm_system.addForce(custom)

            for i in range(self.universe.n_atoms):
                charge, sigma, epsilon = nonbonded.getParticleParameters(i)
                custom.addParticle([sigma, epsilon])

        for i in range(self.universe.n_atoms):
            for j, dist in nx.single_source_shortest_path_length(
                self.bond_graph,
                i,
                cutoff=len(self.nonbonded_scaling),
            ).items():
                if j < i or dist == 0:
                    continue
                # scale nonbonded interaction when atoms are connected
                # by a specific number of bonds away
                scale_q, scale_sigma, scale_eps = self.nonbonded_scaling[dist - 1]
                q_i, sig_i, eps_i = nonbonded.getParticleParameters(i)
                q_j, sig_j, eps_j = nonbonded.getParticleParameters(j)

                charge = scale_q * (q_i * q_j)
                if self.nonbonded_combining == CombiningRules.GEOMETRIC:
                    sigma = scale_sigma * (sig_i * sig_j) ** 0.5
                    epsilon = scale_eps * (eps_i * eps_j) ** 0.5
                    custom.addExclusion(i, j)
                else:
                    sigma = scale_sigma * ((sig_i + sig_j) / 2)
                    epsilon = scale_eps * (eps_i * eps_j) ** 0.5

                nonbonded.addException(i, j, charge, sigma, epsilon, replace=True)

        if self.nonbonded_combining != CombiningRules.LORENTZBERTHLOT:
            for i in range(self.universe.n_atoms):
                charge, sigma, epsilon = nonbonded.getParticleParameters(i)
                nonbonded.setParticleParameters(i, charge, sigma, 0.0)

    def clear_forces_and_constraints(self):
        """Clear the OpenMM force fields and constraints from the OpenMM system."""
        for i in reversed(range(self.openmm_system.getNumForces())):
            self.openmm_system.removeForce(i)
        for i in reversed(range(self.openmm_system.getNumConstraints())):
            self.openmm_system.removeConstraint(i)

    def setup_simulation(
        self,
        *,
        openmm_platform: str | None = None,
        openmm_properties: dict | None = None,
        **settings: Any,
    ) -> None:
        """Set up the openmm simulation.

        Parameters
        ----------
        settings**
            Some settings which are used to set up the openmm
            simulation object.
        """
        self.temperature = float(settings.get("temperature"))
        self.openmm_ensembles = settings.get("openmm_ensembles", self.openmm_ensembles)
        if len(self.openmm_ensembles) < 2:
            raise ValueError(
                "openmm_equilibration needs at least two ensemble "
                "settings one for equilibration and one for production."
            )

        compound_integrator = self.create_compound_integrator()

        if openmm_platform is not None:
            openmm_platform = mm.Platform.getPlatform(openmm_platform)

        self.openmm_simulation = Simulation(
            Topology(),
            self.openmm_system,
            compound_integrator,
            openmm_platform,
            openmm_properties,
        )

        positions = np.array([atom.position for atom in self.universe.atoms]) * unit.angstrom
        self.openmm_simulation.context.setPositions(positions)

        if any(np.any(atom.velocity) for atom in self.universe.atoms):
            self.openmm_simulation.context.setVelocities(
                np.array([atom.velocity for atom in self.universe.atoms])
                * unit.angstrom
                / unit.picosecond,
            )
        else:
            self.openmm_simulation.context.setVelocitiesToTemperature(
                self.temperature * unit.kelvin,
            )

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

    def create_compound_integrator(self) -> mm.Integrator:
        """Create an OpenMM CompoundIntegrator for the equilibration
        and production. The last integrator in the compound integrator
        will be used for in the production stage.

        Returns
        -------
        mm.CompoundIntegrator
            The OpenMM CompoundIntegrator.
        """
        temperature = self.temperature * unit.kelvin
        time_step = float(self.time_step) * unit.femtoseconds
        compound_integrator = mm.CompoundIntegrator()

        for settings in self.openmm_ensembles:
            integrator = settings["integrator"].lower()
            if integrator == "verlet":
                compound_integrator.addIntegrator(mm.VerletIntegrator(time_step))
            elif integrator == "langevin":
                compound_integrator.addIntegrator(
                    mm.LangevinIntegrator(temperature, settings["frictionCoeff"], time_step)
                )
            elif integrator == "langevinmiddle":
                compound_integrator.addIntegrator(
                    mm.LangevinMiddleIntegrator(temperature, settings["frictionCoeff"], time_step)
                )
            elif integrator == "nosehoover":
                compound_integrator.addIntegrator(
                    mm.NoseHooverIntegrator(
                        temperature,
                        settings["collisionFrequency"],
                        time_step,
                        settings.get("chainLength", 3),
                        settings.get("numMTS", 3),
                        settings.get("numYoshidaSuzuki", 7),
                    )
                )
            else:
                raise ValueError(f"Integrator {integrator} not recognised or not implemented.")

        return compound_integrator

    def add_barostat(self, settings: dict):
        """Add a barostat.

        Parameters
        ----------
        settings : dict
            A dictionary of barostat settings.
        """
        name = settings["barostat"].lower()
        if name == "montecarloflexible":
            barostat = mm.MonteCarloFlexibleBarostat(
                settings["defaultPressure"],
                self.temperature * unit.kelvin,
                settings.get("frequency", 25),
                settings.get("scaleMoleculesAsRigid", True),
            )
        elif name == "montecarlo":
            barostat = mm.MonteCarloBarostat(
                settings["defaultPressure"],
                self.temperature * unit.kelvin,
                settings.get("frequency", 25),
            )
        elif name == "montecarloanisotropic":
            barostat = mm.MonteCarloAnisotropicBarostat(
                settings["defaultPressure"],
                self.temperature * unit.kelvin,
                settings.get("scaleX", True),
                settings.get("scaleY", True),
                settings.get("scaleZ", True),
                settings.get("frequency", 25),
            )
        else:
            raise ValueError(f"Barostat {name} not recognised or not implemented.")
        self.openmm_system.addForce(barostat)
        self.openmm_simulation.context.reinitialize(preserveState=True)

    def remove_barostat(self):
        """Remove all barostats."""
        for i in reversed(range(self.openmm_system.getNumForces())):
            force = self.openmm_system.getForce(i)
            if isinstance(
                force,
                (
                    mm.MonteCarloFlexibleBarostat,
                    mm.MonteCarloBarostat,
                    mm.MonteCarloAnisotropicBarostat,
                ),
            ):
                self.openmm_system.removeForce(i)
        self.openmm_simulation.context.reinitialize(preserveState=True)

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
                self.openmm_simulation.context.setVelocitiesToTemperature(
                    self.temperature * unit.kelvin,
                )
                for i, settings in enumerate(self.openmm_ensembles[:-1]):
                    self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(i)

                    if "barostat" in settings:
                        self.add_barostat(settings["barostat"])

                    n_steps = settings.get("n_steps", n_steps)
                    if isinstance(n_steps, int):
                        self.openmm_simulation.step(n_steps)
                    elif isinstance(n_steps, tuple) and len(n_steps) == 5:
                        self.autoequilibrate(*n_steps)
                    else:
                        raise ValueError(f"n_steps setting {n_steps} is not valid")

                    self.remove_barostat()

            except mm.OpenMMException as e:
                self.remove_barostat()
                LOGGER.warning(f"OpenMM exception during equilibration: {e}")
                raise MDEngineError(f"OpenMM exception during equilibration: {e}") from e

        else:
            settings = self.openmm_ensembles[-1]
            if "barostat" in settings:
                self.add_barostat(settings["barostat"])

            self.compact_trajectory = CompactTrajectory()
            self.compact_trajectory.preAllocate(n_steps=n_steps, n_atoms=sum(self.real_atom))
            reporter = CompactTrajectoryReporter(
                self.compact_trajectory,
                self.traj_step,
                n_steps,
                np.array(self.real_atom),
            )
            self.openmm_simulation.reporters.append(reporter)
            self.openmm_simulation.context.getIntegrator().setCurrentIntegrator(
                len(self.openmm_ensembles) - 1
            )
            self.openmm_simulation.currentStep = 0
            self.openmm_simulation.context.setTime(0.0)
            state = self.openmm_simulation.context.getState(
                getPositions=True,
                enforcePeriodicBox=True,
            )
            reporter.report(self.openmm_simulation, state)
            try:
                self.openmm_simulation.step(n_steps)
            except mm.OpenMMException as e:
                LOGGER.warning(f"OpenMM exception during production run: {e}")
                raise MDEngineError(f"OpenMM exception during production run: {e}") from e
            finally:
                self.remove_barostat()
                self.openmm_simulation.reporters.clear()

    def autoequilibrate(
        self,
        ensemble: str,
        max_steps: int,
        eq_steps: int,
        window_size: int,
        tolerance: float,
    ):
        """Runs MD until certain properties have become stationary
        defined by the KPSS test.


        Parameters
        ----------
        ensemble : str
            The ensemble of the system we are equilibrating should be
            either NPT, NVT or NVE. This defines what system properties
            to monitor for the KPSS test.
        max_steps : int
            Max number of MD steps to run before exiting with an error.
        eq_steps : int
            Number of MD steps to run per iteration.
        window_size : int
            Size of the window used to run the KPSS test on.
        tolerance : float
            Tolerance used to define when the property stationary or not.
        """

        def property_is_stationary(values: list) -> bool:
            """KPSS test on system properties, see PR #1298.

            Parameters
            ----------
            values : array_like, 1d
                List of floats of the system properties to run the KPSS test
                on. KPSS test will run on a specific number of values from
                the end defined by window_size.
            """
            vals = values[-window_size:]
            if np.all(np.isfinite(vals)):
                results = kpss(vals, regression="c")
                # results[1] is the p-value from the test
                # we base our tolerance on the p-value, where the alternative hypothesis
                # for KPSS is "NOT stationary" - statsmodels also never gives a p above
                # 0.1 as it doesn't hold critical values above that point. so for 0.05
                # tolerance, we are asking that p be greater than 0.95
                return results[1] > 0.1 - tolerance
            return False

        reporter = PropertyReporter()
        self.openmm_simulation.reporters.append(reporter)
        self.openmm_simulation.step(window_size)
        for _ in range(eq_steps, max_steps + 1, eq_steps):
            self.openmm_simulation.step(eq_steps)
            volumes = reporter.volumes
            temperatures = reporter.temperatures
            energies = reporter.total_energies
            if ensemble == "NPT":
                if all(property_is_stationary(values) for values in [volumes, temperatures]):
                    break
            elif ensemble == "NVT" and property_is_stationary(temperatures):
                break
            elif ensemble == "NVE" and property_is_stationary(energies):
                break
            else:
                ValueError(f"Ensemble {ensemble} not recognised or not supported.")
        else:
            # the equilibration failed. Let's continue anyway, if it only
            # happens once or twice it should be ok since the force
            # field parameters are probably going to be bad anyway
            # if this happens often then the user will see the warnings
            # and will need to adjust the setting e.g. force field parameter
            # search space
            print(
                f"{ensemble} ensemble auto-equilibration has failed after "
                f"{len(reporter.volumes)} equilibration steps. Continuing to "
                f"the next stage anyway. Please adjust your equilibration "
                f"settings so equilibration can be obtained or parameter "
                f"search space as particularly troublesome force field parameters "
                f"were used."
            )
            self.openmm_simulation.reporters.clear()
            return

        print(
            f"{ensemble} ensemble auto-equilibration has detected stability after {len(reporter.volumes)} equilibration steps."
        )
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
        real_atoms = [atom for atom in self.universe.atoms if not isinstance(atom, AverageSite3P)]
        self.compact_trajectory.atom_types = [atom.atom_type for atom in real_atoms]
        atom_elements = {atom.atom_type: atom.element for atom in real_atoms}
        atom_masses = {atom.atom_type: atom.mass for atom in real_atoms}
        self.compact_trajectory.labelAtoms(atom_elements, atom_masses)
        self.compact_trajectory.setCharge([atom.charge for atom in real_atoms])
        self.compact_trajectory.postProcess()
        return self.compact_trajectory

    def update_parameters(self) -> None:
        """Updates the ``OpenMMEngine`` force field parameters."""
        self.clear_forces_and_constraints()
        self.change_openmm_force_field_and_constraints()
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
        self.openmm_simulation.context.setVelocitiesToTemperature(self.temperature * unit.kelvin)

    def eval(self, variable: str):
        raise NotImplementedError


class PropertyReporter:
    def __init__(self):
        """Reporter which saves MD properties"""
        self.volumes = []
        self.temperatures = []
        self.total_energies = []

    def report(self, simulation: Simulation, state: mm.State):
        """Save the simulation properties.

        Parameters
        ----------
        simulation : Simulation
            The openmm simulation object.
        state : mm.State
            The openmm state object.
        """
        # currently MDMC can only deal with orthorhombic lattices
        a, b, c = state.getPeriodicBoxVectors()
        a = a.value_in_unit(unit.angstrom)[0]
        b = b.value_in_unit(unit.angstrom)[1]
        c = c.value_in_unit(unit.angstrom)[2]
        self.volumes.append(a * b * c)

        temperature = (
            2 * (state.getKineticEnergy() / (3 * unit.MOLAR_GAS_CONSTANT_R))
        ).value_in_unit(unit.kelvin)
        self.temperatures.append(temperature)

        self.total_energies.append(
            (state.getKineticEnergy() + state.getPotentialEnergy()).value_in_unit(
                unit.kilojoules_per_mole
            )
        )

    def describeNextReport(self, simulation):
        return 1, False, False, False, True


class CompactTrajectoryReporter:
    def __init__(
        self,
        compact_trajectory: CompactTrajectory,
        report_interval: int,
        n_steps: int,
        real_atom: np.ndarray,
    ):
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
        real_atom : np.ndarray
            An array of bools, true if the atom is not a dummy atom.
        """
        self.compact_trajectory = compact_trajectory
        self.report_interval = report_interval
        self.n_steps = n_steps
        self.real_atom = real_atom

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
            positions=np.array(positions)[self.real_atom],
        )
        self.compact_trajectory.setDimensions(np.array([a, b, c]), step_num=step)

    def describeNextReport(self, simulation: Simulation):
        steps = self.report_interval - simulation.currentStep % self.report_interval

        if simulation.currentStep + steps >= self.n_steps:
            return steps, False, False, False, False

        return steps, True, False, False, False

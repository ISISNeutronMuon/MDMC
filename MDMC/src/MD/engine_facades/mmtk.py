"""Facade for MMTK MD engine

AUTHOR :    Thomas Farmer        START DATE :    2018-5-16 11:07:19"""

import weakref
from tempfile import TemporaryFile

from MDMC.src.MD.engine_facades.facade import MDEngine
from MDMC.src.MD.simulation import Shape
from MDMC.src.MD.force_fields import SPCE
from MDMC.src.MD.structural_units import Dispersion

# TODO: import other modules that need to be wrapped

import MMTK
from MMTK.ForceFields import SPCEFF
from MMTK.Dynamics import VelocityVerletIntegrator, VelocityScaler, \
                            TranslationRemover
from MMTK.Minimization import SteepestDescentMinimizer, \
                                ConjugateGradientMinimizer
from Scientific._vector import Vector
# from MMTK import ChemicalObjects,MMTK.Universe
# from MMTK.Trajectory import Trajectory
# from MMTK.Minimization import SteepestDescentMinimizer
# from MMTK.Dynamics import VelocityVerletIntegrator, VelocityScaler, \
#                             TranslationRemover


UNIVERSE_PBC = {Shape.infinite:MMTK.Universe.InfiniteUniverse,
                Shape.orthorhombic:MMTK.Universe.OrthorhombicPeriodicUniverse,
                Shape.cubic:MMTK.Universe.CubicPeriodicUniverse}
UNIVERSE_FF = {SPCE:SPCEFF.SPCEForceField}
UNIVERSE_INT = {'velocity_verlet':VelocityVerletIntegrator}
UNIVERSE_MINIM = {'steepest_descent':SteepestDescentMinimizer,
                    'conjugate_gradient_minimizer':ConjugateGradientMinimizer}


class MMTKEngine(MDEngine):

    """
    Facade for MMTK API

    Implements simple methods in abstract base class facade.
    """

    # TODO: Enable different universe types
    def setup_universe(self, universe, **settings):
        self._MDMC_universe = weakref.ref(universe)
        self.universe = MMTKCubicUniverse(self.MDMC_universe,
            **settings)
        self.build_configuration()

    def setup_simulation(self, universe, **settings):

        """
        Creates a time integrator and sets relevant simulation parameters

        Call MMTK integrator corresponding to the integrator setting.
        Initialize velocities to Boltzmann distribution based on temperature,
        defaulting to 300 K if no temperature is specified. Set integrator to
        scale velocities to this temperature as it runs, with default allowed
        variation in temperature of 10 K.  MMTK integrator takes time steps in
        ps (although defaults to 1 fs).

        If a minimizer is specified, it will run before the integrator. The step
        size and number of steps of the minimizer can also be passed in
        **settings, with defaults of 100 and 0.05 AA.

        MMTK.Trajectory.Trajectory() is called with a temporary file.
        """

        # TODO: Add in additional actions. Change so that TranslationRemover can be deselected
        temperature = settings.get('temperature',300) * MMTK.Units.K
        temperature_variation = 10. * MMTK.Units.K

        time_step = settings.get('time_step',1) * MMTK.Units.fs

        self._set_trajectory_output()

        self.universe.initializeVelocitiesToTemperature(temperature)
        actions = [self.trajectory_output, TranslationRemover(),
                    VelocityScaler(temperature,temperature_variation)]
        self.integrator = UNIVERSE_INT[settings['integrator']](self.universe,
            delta_t=time_step, actions = actions)
        if 'minimizer' in settings:
            self.minimizer = UNIVERSE_MINIM[settings['minimizer']](
                self.universe,
                step_size = settings.get('minimizer_step_size',0.05))
            self.minimizer_steps = settings.get('minimizer_steps',100)

    def run(self, n_steps):

        """
        Minimizes (if provided with a minimizer) and integrates the universe
        """

        if hasattr(self,'minimizer'):
            self.minimizer(steps = self.minimizer_steps)
        self.integrator(steps = n_steps)

    def _set_trajectory_output(self):
    # TODO: Consider if a SpooledTemporaryFile would be more appropriate
        trajectory_file = TemporaryFile()
        self.trajectory = MMTK.Trajectory.Trajectory(self.universe,
            trajectory_file.name, mode = 'w')
        self.trajectory_output = MMTK.Trajectory.TrajectoryOutput(
            self.trajectory)

    @property
    def MDMC_universe(self):
        return self._MDMC_universe()

    def build_configuration(self):

        """
        Fills the MMTK universe from MDMC universe configuration and topology

        Only adds atoms in molecules to MMTK universe, as MMTK does not define
        force fields on individual atoms (except about gases, excluding He)
        """

        # TODO: Add additional parameters that can be passed to an MMTK molecule
        for molecule in self.MDMC_universe.molecule_list:
            m = MMTKMolecule(molecule)
            self.universe.addObject(m)


# TODO: Take out self.MDMC_obj definition into mixin class
# TODO: Expand to other universes
class MMTKCubicUniverse(MMTK.Universe.CubicPeriodicUniverse):

    def __init__(self, universe, **settings):
        self._MDMC_obj = weakref.ref(universe)

        if universe.shape is Shape.cubic:
            dims = universe.dims[0]
        else:
            dims = universe.dims
        if universe.force_field is None:
            super(MMTKCubicUniverse,self).__init__(universe.dims[0])
        else:
            super(MMTKCubicUniverse,self).__init__(universe.dims[0],
                UNIVERSE_FF[type(universe.force_field)](
                settings.get('lj_options',None),
                settings.get('es_options',None)))
        self.assign_lj_parameters()

    @property
    def MDMC_obj(self):
        return self._MDMC_obj()

    def update_MDMC_obj(self):
        pass

    def assign_lj_parameters(self):
        self.forcefield().nonbonded.dataset.ljParameters = \
            self.lj_parameters_closure(self.MDMC_obj.element_dict)

    def lj_parameters_closure(self, element_dict):

        """
        LJ Parameters closure returns function equivalent to
        MMTK SPCEParameters.ljParameters method, which is where LJ parameters
        are hard coded. Parameters are returned (as a tuple) from first
        Dispersion interaction of MDMC atom of same element.
        """

        parameters = {element:self.get_interaction_parameters(atom,
            Dispersion) for element, atom in element_dict.items()}

        def lj_parameters(type):
            try:
                return parameters[type]
            except:
                raise ValueError('Unknown atom type' + type)

        return lj_parameters

    # TODO: Change hard coded return (if atom has no interaction of that type)
    def get_interaction_parameters(self, element, interaction_type):
        try:
            parameters = tuple(next(interaction.function.params.values() for
                interaction in element.interaction_list() if
                isinstance(interaction,interaction_type)))

            # TODO: MMTK takes three parameters for LJ, with the third commonly being hard coded to 0 - determine why and account for this better than below
            if interaction_type == Dispersion:
                return parameters + (0,)
            else:
                return parameters
        except StopIteration:
            if interaction_type == Dispersion:
                return (0., 0., 0)
            else:
                return (0., 0.)


class MMTKAtom(MMTK.ChemicalObjects.Atom):

    def __init__(self, atom, atom_spec, **properties):
        self._MDMC_obj = weakref.ref(atom)
        super(MMTKAtom,self).__init__(atom_spec, **properties)

    @property
    def MDMC_obj(self):
        return self._MDMC_obj()

    # TODO: MMTK doesn't support spce for individual atoms (or in fact for Amber forcefields)
    # def assign_charge(self):
    #     self.topLevelChemicalObject().spce_charge[
    #         self.topLevelChemicalObject().getReference(self)] = \
    #         self.MDMC_obj.charge

    def assign_lj_parameters(self):
        pass

    def update_MDMC_obj(self):
        pass

# TODO: Modify so if molecule.name = None, it can be determined from MMTK database based on atoms and bonds
class MMTKMolecule(MMTK.ChemicalObjects.Molecule):

    def __init__(self, molecule, **properties):
        self._MDMC_obj = weakref.ref(molecule)
        super(MMTKMolecule,self).__init__(molecule.name,
            position = Vector(molecule.position), **properties)

    @property
    def MDMC_obj(self):
        return self._MDMC_obj()

    # TODO: Find a better way of matching atoms in Molecules in MDMC and MMTK
    # TODO: Only works for spce
    def assign_charge(self):
        for atom in self.atomList():
            for MDMC_atom in self.MDMC_obj().atom_list:
                if atom.type.symbol == MDMC_atom.element:
                    atom.topLevelChemicalObject().spce_charge[
                        atom.topLevelChemicalObject().getReference(atom)] = \
                        MDMC_atom.charge
                    break

    def update_MDMC_obj(self):
        pass

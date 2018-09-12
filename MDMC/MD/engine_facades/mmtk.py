"""Facade for MMTK MD engine

AUTHOR :    Thomas Farmer        START DATE :    2018-5-16 11:07:19"""

import weakref
from tempfile import TemporaryFile

from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.MD.simulation import Shape
from MDMC.MD.force_fields import SPCE
import MDMC.MD.structural_units as MDMCs
import MDMC.trajectory_analysis.trajectory as MDMCt

# TODO: import other modules that need to be wrapped

import MMTK
from MMTK.ForceFields import SPCEFF
from MMTK.Dynamics import VelocityVerletIntegrator, VelocityScaler, \
                            TranslationRemover
from MMTK.Minimization import SteepestDescentMinimizer, \
                                ConjugateGradientMinimizer
from Scientific._vector import Vector


UNIVERSE_PBC = {Shape.infinite:MMTK.Universe.InfiniteUniverse,
                Shape.orthorhombic:MMTK.Universe.OrthorhombicPeriodicUniverse,
                Shape.cubic:MMTK.Universe.CubicPeriodicUniverse}
UNIVERSE_FF = {SPCE:SPCEFF.SPCEForceField}
UNIVERSE_INT = {'velocity_verlet':VelocityVerletIntegrator}
UNIVERSE_MINIM = {'steepest_descent':SteepestDescentMinimizer,
                  'conjugate_gradient_minimizer':ConjugateGradientMinimizer}


class MMTKEngine(MDEngine):

    """
    Facade for MMTK
    """

    # TODO: Enable different universe types
    def setup_universe(self, universe, **settings):

        self.universe = MMTKCubicUniverse(universe, **settings)
        self.build_configuration(universe)

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
        self.temperature = settings.get('temperature', 300) * MMTK.Units.K
        self.temperature_variation = 10. * MMTK.Units.K
        self.time_step = settings.get('time_step', 1) * MMTK.Units.fs
        self.integrator_type = UNIVERSE_INT[settings['integrator']]

        if 'minimizer' in settings:
            self.minimizer = UNIVERSE_MINIM[settings['minimizer']](
                self.universe,
                step_size=settings.get('minimizer_step_size', 0.05))
        else:
            self.universe.initializeVelocitiesToTemperature(self.temperature)

    def minimize(self, n_steps):

        self.minimizer(steps = n_steps)
        self.universe.initializeVelocitiesToTemperature(self.temperature)

    def run(self, n_steps):

        # Including the trajectory and integrator setup before running resets
        # the trajectory before each run
        self._set_trajectory_output()
        actions = [self.trajectory_output, TranslationRemover(),
                   VelocityScaler(self.temperature, self.temperature_variation)]
        self.integrator = self.integrator_type(self.universe,
                                               delta_t=self.time_step,
                                               actions=actions)

        self.integrator(steps = n_steps)

    def _set_trajectory_output(self):

        """
        Creates a temporary file in which to output MMTK trajectories
        """

        # TODO: Consider if a SpooledTemporaryFile would be more appropriate
        trajectory_file = TemporaryFile()
        self.trajectory = MMTK.Trajectory.Trajectory(self.universe,
                                                     trajectory_file.name,
                                                     mode='w')
        self.trajectory_output = MMTK.Trajectory.TrajectoryOutput(
            self.trajectory)

    def update_parameters(self):

        self.universe.assign_lj_parameters()
        self.universe.assign_bond_parameters()
        self.universe.assign_bond_angle_parameters()
        for mol in self.universe:
            mol.assign_charge()

    def build_configuration(self, MDMC_universe):

        """
        Fills the MMTK universe from MDMC universe configuration and topology

        Only adds atoms in molecules to MMTK universe, as MMTK does not define
        force fields on individual atoms (except about noble gases, excluding
        He)
        """

        # TODO: Add additional parameters that can be passed to an MMTK molecule
        for molecule in MDMC_universe.molecule_list:
            m = MMTKMolecule(molecule)
            self.universe.addObject(m)

    # TODO: Unite convert trajectory methods once testing is complete
    def convert_trajectory(self):

        return convert_trajectory(self.trajectory)

class MMTKCubicUniverse(MMTK.Universe.CubicPeriodicUniverse):

    def __init__(self, universe, **settings):

        """
        As the base class defines __setattr__, self.MDMC_universe cannot have a
        setter method which sets self._MDMC_universe as a weakref. The
        consequence of this is that self.MDMC_universe cannot be set after
        initialization.
        """

        self._MDMC_universe = weakref.ref(universe)
        dims = self.MDMC_universe.dims[0] / 10.
        if universe.force_fields is None:
            super(MMTKCubicUniverse, self).__init__(dims)
        else:
            super(MMTKCubicUniverse, self).__init__(
                dims,
                UNIVERSE_FF[type(universe.force_fields)](
                    settings.get('lj_options', None),
                    settings.get('es_options', None)))
        self.assign_lj_parameters()

    @property
    def MDMC_universe(self):

        return self._MDMC_universe()

    @MDMC_universe.setter
    def MDMC_universe(self, universe):

        self._MDMC_universe = weakref.ref(universe)

    def update_MDMC_universe(self):

        raise NotImplementedError

    def assign_lj_parameters(self):

        """
        Sets the MMTK function which defines the LJ parameters equal to the
        function returned by self.lj_parameters_closure.
        """

        self.forcefield().nonbonded.dataset.ljParameters = \
            self.lj_parameters_closure(self.MDMC_universe.element_dict)

    def lj_parameters_closure(self, element_dict):

        """
        Returns:
        Function equivalent to MMTK SPCEParameters.ljParameters method, which is
        where LJ parameters are hard coded in MMTK.
        """

        parameters = {element:self.get_interaction_parameters(atom,
                                                              MDMCs.Dispersion)
                      for element, atom in element_dict.items()}

        def lj_parameters(atom_type):
            try:
                return parameters[atom_type]
            except:
                raise ValueError('Unknown atom type' + atom_type)

        return lj_parameters

    # TODO: Change hard coded return (if atom has no interaction of that type)
    def get_interaction_parameters(self, element, interaction_type):

        """
        Gets the parameters by finding the first interaction with the specified
        interaction type.  If no matching interaction type is found then all
        parameters are set to 0.

        MMTK requires a third parameter for LJ which appears to always be 0. As
        the LJ force field normally only takes two parameters, this third
        parameter will always be hard coded to 0.

        Returns:
        A tuple from first Dispersion interaction of MDMC atom of same element.
        """

        try:
            parameters = tuple(next(interaction.function.params_values for
                                    interaction in element.interactions if
                                    isinstance(interaction, interaction_type)))

            if interaction_type == MDMCs.Dispersion:
                return parameters + (0,)
            return parameters
        except StopIteration:
            if interaction_type == MDMCs.Dispersion:
                return (0., 0., 0)
            return (0., 0.)


class MMTKAtom(MMTK.ChemicalObjects.Atom):

    def __init__(self, atom, atom_spec, **properties):
        self._MDMC_obj = weakref.ref(atom)
        super(MMTKAtom, self).__init__(atom_spec, **properties)

    @property
    def MDMC_obj(self):
        return self._MDMC_obj()

    # TODO: MMTK doesn't support spce for individual atoms (same for Amber forcefields)
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
        position = coordinate_transform(molecule.position,
                                        molecule.universe.dims / 10.,
                                        from_MDMC=True)
        super(MMTKMolecule, self).__init__(molecule.name,
                                           position=Vector(position),
                                           **properties)

    @property
    def MDMC_obj(self):
        return self._MDMC_obj()

    # TODO: Find a better way of matching atoms in Molecules in MDMC and MMTK
    # TODO: Only works for spce
    def assign_charge(self):
        for atom in self.atomList():
            for MDMC_atom in self.MDMC_obj.atom_list:
                if atom.type.symbol == MDMC_atom.element:
                    atom.topLevelChemicalObject().spce_charge[
                        atom.topLevelChemicalObject().getReference(atom)] = \
                        MDMC_atom.charge
                    break

    def update_MDMC_obj(self):
        pass


# TODO: Update this so that it can form an association with specific atoms and also creates molecules not just atoms
def convert_trajectory(MMTK_trajectory, **kwargs):

    """
    Builds an MDMC trajectory from an MMTK trajectory.

    Assumes that there is no change in the number/types of atom in the universe.

    Arguments:
    MMTK_trajectory: An MMTK trajectory
    """

    # This currently exists as a separate function so that saved MMTK trajectories can be tested
    # For the same reason the ability to filter the trajectories is provided

    universe_dims = MMTK_trajectory.universe.cellParameters()

    # List of atom element and masses as ordered in configuration
    atom_list = [(atom.type.symbol, atom.mass()) for atom in
                 MMTK_trajectory.universe.atomList()]

    configurations = []
    if kwargs.get('slice'):
        for i in range(kwargs.get('slice').get('start'),
                       kwargs.get('slice').get('stop'),
                       kwargs.get('slice').get('step')):
            configurations.append(convert_configuration(MMTK_trajectory[i],
                                                        universe_dims,
                                                        atom_list))
    else:
        for MMTK_frame in MMTK_trajectory:
            configurations.append(convert_configuration(MMTK_frame,
                                                        universe_dims,
                                                        atom_list))

    return MDMCt.Trajectory(*configurations)


def convert_configuration(MMTK_frame, uni_dims, atom_list=None):

    """
    Builds an MDMC configuration from an MMTK configuration.

    Arguments:
    MMTK_frame - A frame of an MMTK trajectory ir a configuration
    uni_dims - a vector of the MDMC universe dimensions
    atom_list - a list of the atoms in the configuration
    """

    if atom_list is None:
        atom_list = [(atom.type.symbol, atom.mass()) for atom in
                     MMTK_frame.universe.atomList()]

    atoms = []
    for index in range(len(MMTK_frame['configuration'])):
        symbol = atom_list[index][0]
        mass = atom_list[index][1]
        MMTK_position = MMTK_frame['configuration'].__dict__['array'][index]
        position = coordinate_transform(MMTK_position,
                                        uni_dims,
                                        from_MDMC=False)
        atoms.append(MDMCs.Atom(symbol, position=position, mass=mass))
    return MDMCt.TemporalConfiguration(MMTK_frame['time'], *atoms)


def coordinate_transform(coordinates, uni_dims, from_MDMC=True):

    """
    Transforms between MDMC and MMTK coordinate systems

    MDMC coordinates are only in a positive quadrant and are in Angstroms,
    whereas MMTK coordinates are in all four quadrants are in nm.

    Arguments:
    coordinates - the coordinate vector
    uni_dims - a vector of the MMTK universe dimensions, equivalent to
    MMTK universe.cellParameters()
    from_MDMC - If True convert from MDMC coordinates to MMTK, if False
    convert from MMTK coordinates to MDMC.

    Returns:
    Vector of transformed coordinates
    """

    if from_MDMC:
        coord = coordinates / 10. - uni_dims / 2.
    else:
        coord = (coordinates + uni_dims / 2.) * 10.

    return coord

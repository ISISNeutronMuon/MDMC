"""Facade for MMTK MD engine

AUTHOR :    Thomas Farmer        START DATE :    2018-5-16 11:07:19"""

from tempfile import TemporaryFile
import weakref

import MMTK
from MMTK import Units
from MMTK.Dynamics import VelocityVerletIntegrator, VelocityScaler, \
                            TranslationRemover, BarostatReset
from MMTK.ForceFields import SPCEFF
from MMTK.Minimization import SteepestDescentMinimizer, \
                                ConjugateGradientMinimizer
from MMTK.Environment import AndersenBarostat, NoseThermostat
from MMTK.Trajectory import StandardLogOutput
import numpy as np
from Scientific._vector import Vector

from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.MD.force_fields import SPCE
from MDMC.MD.simulation import Shape
import MDMC.MD.structural_units as MDMCs
import MDMC.trajectory_analysis.trajectory as MDMCt


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

    def setup_simulation(self, **settings):

        """
        Creates a time integrator and sets relevant simulation parameters

        Calls MMTK integrator corresponding to the integrator setting.
        Initializes velocities to Boltzmann distribution based on temperature,
        defaulting to 300 K if no temperature is specified. The temperature is
        scaled to this value during equilibration but not during production
        runs.

        MMTK.Trajectory.Trajectory() is called with a temporary file.

        Arguments:
        Settings:
        temperature - float temperature in units of K (default 300 K)
        temperature_variation - float maximum temperature before scaling in K
        (default 10 K)
        traj_step - the number of simulation steps between each trajectory
        output
        time_step - float time step in units of fs (default 1 fs)
        minimizer_step_size - float minimizer distance step in units of AA
        (default 0.05 AA)
        pressure - float pressure in units of atm
        """

        self.temperature = settings.get('temperature', 300) * MMTK.Units.K
        self.temperature_variation = (settings.get('temperature_variation', 10.)
                                      * MMTK.Units.K)
        self.time_step = settings.get('time_step', 1) * MMTK.Units.fs
        self.integrator_type = UNIVERSE_INT[settings['integrator']]
        self.traj_step = settings['traj_step']
        self.rigid = settings.get('rigid', False)
        self.threads = settings.get('threads', 1)
        self.thermostat = settings.get('thermostat', None)
        self.pressure = settings.get('pressure', None)

        if 'minimizer' in settings:
            self.minimizer = UNIVERSE_MINIM[settings['minimizer']](
                self.universe,
                step_size=settings.get('minimizer_step_size', 0.05)*Units.Ang,
                threads=self.threads)
        else:
            self.universe.initializeVelocitiesToTemperature(self.temperature)

    def minimize(self, n_steps):

        self.minimizer(steps = n_steps, step_size = 0.05*Units.Ang)
        self.universe.initializeVelocitiesToTemperature(self.temperature)

    def run(self, n_steps, equilibration=False):

        # Including the trajectory and integrator setup before running resets
        # the trajectory before each run
        self._set_trajectory_output()
        actions = [self.trajectory_output, TranslationRemover(), StandardLogOutput()]

        if self.thermostat:
            self.universe.thermostat = NoseThermostat(self.temperature,
                                                      relaxation_time= \
                                                          100.*Units.fs)

        if self.pressure:
            self.universe.barostat = AndersenBarostat(self.pressure * Units.atm,
                                                      relaxation_time= \
                                                          100.*Units.fs)
            actions.append(BarostatReset(0, None, 10))

        if self.rigid:
            self.universe.setBondConstraints()

        if equilibration:
            actions.append(VelocityScaler(self.temperature,
                                          self.temperature_variation))


        self.integrator = self.integrator_type(self.universe,
                                               delta_t=self.time_step,
                                               actions=actions,
                                               threads=self.threads)
        self.integrator(steps = n_steps)

    def _set_trajectory_output(self):

        """
        Creates a temporary file in which to output MMTK trajectories
        """

        trajectory_file = TemporaryFile()
        self.trajectory = MMTK.Trajectory.Trajectory(self.universe,
                                                     trajectory_file.name,
                                                     mode='w')
        self.trajectory_output = MMTK.Trajectory.TrajectoryOutput(
            self.trajectory,
            ("time", "energy", "thermodynamic", "configuration"),
            0, None, self.traj_step)

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
        force fields on individual atoms (except for noble gases, excluding
        He)
        """

        for molecule in MDMC_universe.molecule_list:
            m = MMTKMolecule(molecule)
            self.universe.addObject(m)

    # TODO: Unite convert trajectory methods once testing is complete
    def convert_trajectory(self):

        return convert_trajectory(self.trajectory,
                                  MDMC_universe=self.universe.MDMC_universe)


class MMTKCubicUniverse(MMTK.Universe.CubicPeriodicUniverse):

    def __init__(self, universe, **settings):

        """
        As the base class defines __setattr__, self.MDMC_universe cannot have a
        setter method which sets self._MDMC_universe as a weakref. The
        consequence of this is that self.MDMC_universe cannot be set after
        initialization.

        Arguments:
        universe - an MDMC universe
        Settings:
        lj_options - Either a float specifying the cutoff in AA or a string
        specifying the cutoff type
        es_options - Either a float specifying the cutoff in AA or a string
        specifying the cutoff type
        """

        self._MDMC_universe = weakref.ref(universe)
        dims = self.MDMC_universe.dims[0] / 10.

        ls = self.parse_ff_option(settings.get('ls_options'))
        es = self.parse_ff_option(settings.get('es_options'))

        if universe.force_fields is None:
            super(MMTKCubicUniverse, self).__init__(dims)
        else:
            super(MMTKCubicUniverse, self).__init__(
                dims,
                UNIVERSE_FF[type(universe.force_fields)](ls, es))
        self.assign_lj_parameters()
        self.assign_bond_parameters()
        self.assign_bond_angle_parameters()

    @property
    def MDMC_universe(self):

        return self._MDMC_universe()

    @MDMC_universe.setter
    def MDMC_universe(self, universe):

        self._MDMC_universe = weakref.ref(universe)

    def update_MDMC_universe(self):

        """
        Updates the positions and velocities of atoms in the MDMC universe
        """

        raise NotImplementedError

    def assign_lj_parameters(self):

        """
        Sets the MMTK function which defines the LJ parameters equal to the
        function returned by self.lj_parameters_closure.
        """

        self._forcefield.nonbonded.dataset.ljParameters = \
            self.lj_parameters_closure(self.MDMC_universe.element_dict)

    def lj_parameters_closure(self, element_dict):

        """
        Arguments:
        element_dict - a dictionary with (element string):(atom of element)
        pairs

        Returns:
        Function equivalent to MMTK SPCEParameters.ljParameters method, which is
        where LJ parameters are hard coded in MMTK.
        """

        parameters = {element:self.get_interaction_parameters(MDMCs.Dispersion,
                                                              element)
                      for element in element_dict.keys()}

        def lj_parameters(atom_type):
            try:
                return parameters[atom_type]
            except:
                raise ValueError('Unknown atom type' + atom_type)

        return lj_parameters

    def assign_bond_parameters(self):

        """
        Sets the MMTK function which defines the bond Parameters equal to the
        function returned by self.bond_parameters_closure.
        """

        self._forcefield.bonded.dataset.bondParameters = \
            self.bond_parameters_closure(self.MDMC_universe.element_dict)

    def bond_parameters_closure(self, element_dict):

        """
        Arguments:
        element_dict - a dictionary with (element string):(atom of element)
        pairs

        Returns:
        Function equivalent to MMTK SPCEParameters.bondParameters method, which
        is where bond parameters are hard coded in MMTK.
        """

        parameters = {}
        for element1 in element_dict.keys():
            for element2 in element_dict.keys():
                elements = sorted([element1, element2])
                parameters[tuple(sorted([element1, element2]))] = \
                    self.get_interaction_parameters(MDMCs.Bond, *elements)

        def bond_parameters(*atom_type):
            try:
                return parameters[tuple(sorted(atom_type))]
            except:
                raise ValueError('Unknown atom combination '
                                 + atom_type)

        return bond_parameters

    def assign_bond_angle_parameters(self):

        """
        Sets the MMTK function which defines the bond Parameters equal to the
        function returned by self.bond_parameters_closure.
        """

        self._forcefield.bonded.dataset.bondAngleParameters = \
            self.bond_angle_parameters_closure(self.MDMC_universe.element_dict)

    def bond_angle_parameters_closure(self, element_dict):

        """
        Arguments:
        element_dict - a dictionary with (element string):(atom of element)
        pairs

        Returns:
        Function equivalent to MMTK SPCEParameters.bondAngleParameters method,
        which is where bond parameters are hard coded in MMTK.
        """

        parameters = {}
        for element1 in element_dict.keys():
            for element2 in element_dict.keys():
                for element3 in element_dict.keys():
                    parameters[tuple([element1, element2, element3])] =\
                        self.get_interaction_parameters(MDMCs.BondAngle,
                                                        element1,
                                                        element2,
                                                        element3)

        def bond_angle_parameters(*atom_type):
            try:
                return parameters[atom_type]
            except:
                raise ValueError('Unknown atom combination '
                                 + atom_type)

        return bond_angle_parameters

    def get_interaction_parameters(self, interaction_type, *elements):

        """
        Gets the parameters by finding the first interaction with the specified
        interaction type.  If no matching interaction type is found then all
        parameters are set to 0.

        MMTK requires a third parameter for LJ which appears to always be 0. As
        the LJ force field normally only takes two parameters, this third
        parameter will always be hard coded to 0.

        Arguments:
        interaction_type - a class which is a subclass of MDMCs.Interaction
        elements - one or more strings specifying an element

        Returns:
        A tuple from first Dispersion interaction of MDMC atom of same element.
        """

        elements = list(elements)
        for interaction in self.MDMC_universe.interactions:
            if isinstance(interaction, interaction_type):
                int_elements = [atom.element for atom
                                in interaction.atom_list]
                if len(elements) == 2:
                    int_elements = sorted(int_elements)
                if int_elements == elements:
                    parameters = interaction.function.params_values
                    # Converting from Angstroms to nm for bonds and LJ
                    if interaction_type == MDMCs.Dispersion:
                        parameters[1] /= 10.
                        return tuple(parameters) + (0,)
                    if interaction_type == MDMCs.Bond:
                        parameters[0] *= 0.1
                        parameters[1] *= 100.
                    # Converting from degrees to radians for bond angles
                    if interaction_type == MDMCs.BondAngle:
                        parameters[0] *= 2. * np.pi / 360.
                    return tuple(parameters)

        if interaction_type == MDMCs.Dispersion:
            return (0., 0., 0.)

        # For bond constraints applied to water, MMTK requires a bond to be
        # defined between the two hydrogen atoms.  This bond only has a defined
        # length and not a strength.
        if interaction_type == MDMCs.Bond and elements == ['H', 'H']:
            return(0.163298086184, 0.)

        return (0., 0.)

    def parse_ff_option(self, option):

        """
        Parses forcefield option, either electrostatic or LJ

        option - a float specifying the cutoff, a string specifying the
        method for calculating the cutoff or None
        """

        if isinstance(option, float):
            return option * Units.Ang
        elif isinstance(option, str):
            return {'method':option}
        elif option is None:
            return option
        else:
            raise TypeError('Invalid forcefield option (es or lj) specified')


class MMTKAtom(MMTK.ChemicalObjects.Atom):

    """
    An MMTK atom with a reference to the correpsonding MDMC atom
    """

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

    """
    An MMTK molecule with a reference to the corresponding MDMC molecule and a
    method for assigning the charge
    """

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

    def assign_charge(self):

        """
        Assigns the charge to each atom in the MMTK molecule based on the
        charges of atoms of the corresponding MDMC molecule

        CURRENTLY ONLY APPLIES TO SPCE CHARGES
        """

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
def convert_trajectory(MMTK_trajectory, **settings):

    """
    Builds an MDMC trajectory from an MMTK trajectory.

    Assumes that there is no change in the number/types of atom in the universe.

    Arguments:
    MMTK_trajectory: An MMTK trajectory
    Settings:
    slice - a slice to be applied to the MMTK trajectory before conversion
    """

    # This currently exists as a separate function so that saved MMTK trajectories can be tested
    # For the same reason the ability to filter the trajectories is provided

    universe_dims = MMTK_trajectory.universe.cellParameters()
    MDMC_universe = settings.get('MDMC_universe', None)

    # List of atom element and masses as ordered in configuration
    atom_list = [(atom.type.symbol, atom.mass()) for atom in
                 MMTK_trajectory.universe.atomList()]

    configurations = []
    slce = settings.get('slice')
    if slce:
        for i in range(slce.start, slce.stop, slce.step):
            configurations.append(convert_configuration(MMTK_trajectory[i],
                                                        universe_dims,
                                                        MDMC_universe,
                                                        atom_list))
    else:
        for MMTK_frame in MMTK_trajectory:
            configurations.append(convert_configuration(MMTK_frame,
                                                        universe_dims,
                                                        MDMC_universe,
                                                        atom_list))

    return MDMCt.Trajectory(*configurations)


def convert_configuration(MMTK_frame, uni_dims, MDMC_universe, atom_list=None):

    """
    Builds an MDMC configuration from an MMTK configuration.

    Arguments:
    MMTK_frame - A frame of an MMTK trajectory ir a configuration
    uni_dims - a vector of the MMTK universe dimensions
    MDMC_universe - an MDMC universe
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
        atom = MDMCs.Atom(symbol, position=position, mass=mass)
        atom.universe = MDMC_universe
        atoms.append(atom)
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

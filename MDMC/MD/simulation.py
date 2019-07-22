"""Module for setting up and running the simulation

 Classes for the simulation box, minimizer and integrator."""

from abc import ABCMeta, abstractproperty
import copy
from collections import defaultdict
from itertools import count, ifilterfalse, product

from enum import Enum
import numpy as np

from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.common import units
from MDMC.common.configurations import (molec_from_dict, molec_list_from_config,
                                        SPCE)
from MDMC.MD.engine_facades.facade_factory import MDEngineFacadeFactory
from MDMC.MD.force_fields.force_field_factory import ForceFieldFactory
from MDMC.trajectory_analysis.trajectory import Configuration


Shape = Enum('Shape', ['cubic', 'orthorhombic', 'infinite',
                       'rhombic_dodecahedron', 'truncated_octahedron'])


class Universe(object):

    """
    Class where configuration and topology are defined

    Parameters
    ----------
    dims : np.array, list, float
        Dimensions of the Universe, in units of Ang. A float can be used for a
        cubic universe.
    shape : enum
        Member of the Shape enum.
    force_field : ForceField
        A force field to apply to the Universe.
    structures : list
        Structures contained in the Universe.

    **settings
        kspace_solver : KSpaceSolver
            The k-space solver to be used for both electrostatic and dispersive
            interactions. If this is passed then no electrostatic_solver or
            dispersive_solver may be passed.
        electrostatic_solver : KSpaceSolver
            The k-space solver to be used for electrostatic interactions.
        dispersive_solver : KSpaceSolver
            The k-space solver to be used for dispersive interactions.
        constraint_algorithm : ConstraintAlgorithm
            The constraint algorithm which will be applied to constrained
            BondedInteractions.

    Attributes
    ----------
    shape : enum
        Member of the Shape enum.
    kspace_solver : KSpaceSolver
        The k-space solver to be used for both electrostatic and dispersive
        interactions.
    electrostatic_solver : KSpaceSolver
        The k-space solver to be used for electrostatic interactions.
    dispersive_solver : KSpaceSolver
        The k-space solver to be used for dispersive interactions.
    constraint_algorithm : ConstraintAlgorithm
        The constraint algorithm which will be applied to constrained
        BondedInteractions.
    """

    def __init__(self, dimensions, shape=Shape.cubic, force_field=None,
                 structures=None, **settings):

        self.shape = shape
        self.dims = dimensions
        self._atom_types = defaultdict(list)
        self._atom_type_interactions = {}
        if structures:
            self.configuration = Configuration(structures)
        else:
            self.configuration = Configuration(universe=self)
        self._bonded_interaction_pairs = set()
        self._nonbonded_interactions = set()
        self.force_fields = force_field

        self.kspace_solver = settings.get('kspace_solver')
        self.electrostatic_solver = settings.get('electrostatic_solver')
        self.dispersive_solver = settings.get('dispersive_solver')
        # kspace_solver is mutually excusive with the other two solver
        # attributes
        if self.kspace_solver and (self.electrostatic_solver or
                                   self.dispersive_solver):
            raise ValueError('No other solver may be passed if kspace_solver is'
                             ' passed')

        self.constraint_algorithm = settings.get('constraint_algorithm')

    # Unit decorator on getter due to operations in setter
    @property
    @unit_decorator_getter(unit=units.LENGTH)
    def dims(self):

        """
        Get or set the dimensions of the Universe

        Raises
        ------
        TypeError
            If the dimensions of a non-cubic Universe are specified with a float
        TypeError
            If a float, list, or np.array are not passed
        ValueError
            If a list or array is not 1d with 3 elements
        Returns
        -------
        np.array
            The dimensions of the Universe
        """

        return self._dims

    @dims.setter
    def dims(self, dims):

        if isinstance(dims, float):
            if self.shape == Shape.cubic:
                self._dims = np.array([dims] * 3)
            else:
                raise TypeError("Only dimensions of cubic Universes can be"
                                " specified with a float")
        elif isinstance(dims, (list, tuple, np.ndarray)):
            if len(dims) == 3:
                self._dims = np.array(dims)
            else:
                raise ValueError("3 dimensions must be specified")
        else:
            raise TypeError("dims must be a float or 3 element list of floats")

    @property
    def interactions(self):

        """
        Get the interactions in the Universe

        Returns
        -------
        list
            The interactions in the Universe
        """

        return self.bonded_interactions + self.nonbonded_interactions

    @property
    def bonded_interactions(self):

        """
        Get the bonded interactions in the Universe

        Returns
        -------
        list
            The bonded interactions in the Universe
        """

        return [pair[0] for pair in self.bonded_interaction_pairs]

    @property
    def nonbonded_interactions(self):

        """
        Get the nonbonded interactions in the Universe

        Returns
        -------
        list
            The nonbonded interactions in the Universe
        """

        return list(self._nonbonded_interactions)

    @property
    def bonded_interaction_pairs(self):

        """
        Get the bonded interactions and the atoms they apply to

        Returns
        -------
        list
            The (interaction, atoms) pairs in the Universe, where atoms is a
            tuple of all atoms for that specific interaction

        Example
        -------
        For an O Atom with two bonds, one to H1 and one to H2::

        >>> print(O.bonded_interaction_pairs)
        [(Bond, (H1, O)), (Bond, (H2, O))]
        """

        # bonded_interaction_pairs is a set to avoid double counting of
        # interactions
        return list(self._bonded_interaction_pairs)

    @property
    def parameters(self):

        """
        Get the parameters of the interactions that exist within the Universe

        Returns
        -------
        set
            The parameters in the Universe
        """

        return set([param for interaction in self.interactions
                    for param in interaction.params])

    @property
    @unit_decorator_getter(unit=units.LENGTH ** 3)
    def volume(self):

        """
        Get the volume of the Universe

        Returns
        -------
        float
            Volume in Ang^3
        """

        return np.prod(self.dims)

    @property
    def element_list(self):

        """
        The elements of the atoms in the Universe.

        Returns
        -------
        list
            The elements in the Universe.
        """

        return [atom.element for atom in self.atom_list]

    @property
    def element_dict(self):

        """
        Get the elements in the Universe and examples atoms for each element

        This is required for MD engines which assign the same potential
        parameters for all identical element types.

        Returns
        -------
        dict
            element:atom pairs, where atom is a single atom of the specified
            element.

        """

        return {atom.element:atom for atom in self.atom_list}

    @property
    def atom_list(self):

        """
        Get a list of the atoms in the Universe

        Returns
        -------
        list
            The atoms in the Universe
        """

        return self.configuration.atom_list

    @property
    def molecule_list(self):

        """
        Get a list of the molecules in the Universe

        Returns
        -------
        list
            The molecules in the Universe
        """

        return self.configuration.molecule_list

    @property
    def structure_list(self):

        """
        Get a list of all structural units that exist in the Universe.  This
        includes all structural units that are a subunit of another structure
        belonging to the universe.


        Returns
        -------
        list
            The structural units in the Universe
        """

        def add_all_parents(unit):

            parent = unit.parent
            parents = [parent]
            if parent is not parent.top_level_structure():
                parents += add_all_parents(parent)
            return parents

        structural_units = []
        for atom in self.atom_list:
            structural_units += add_all_parents(atom)

        structural_units += list(self.atom_list)
        return list(set(structural_units))

    @property
    def force_fields(self):

        """
        Get or set the force fields acting on the Universe

        Returns
        -------
        list
            Force fields that apply to the Universe
        """

        return self._force_fields

    @force_fields.setter
    def force_fields(self, force_field):

        if force_field:
            self._force_fields = ForceFieldFactory.create_force_field(
                force_field)
        else:
            self._force_fields = None

    @property
    def atom_types(self):

        """
        Get the atom types of atoms in the Universe

        Returns
        -------
        list
            The atom types in the Universe
        """

        return self._atom_types

    @property
    def atom_type_interactions(self):

        """
        Get the atom types and the interactions for each atom type

        Returns
        -------
        dict
            atom_type:interactions pairs where atom_type is a int specifying the
            atom type and interactions is a list of interactions acting on that
            atom_type
        """

        return self._atom_type_interactions

    def _update_atom_types(self, atom):

        """
        Adds the atom to atom_types dictionary

        Parameters
        ----------
        atom : Atom
            An Atom object to add to the atom_types dictionary
        """

        inter_key = (atom.element, ) + tuple(sorted(atom.interactions))
        if atom.atom_type:
            atom_type = atom.atom_type
            if atom_type not in self.atom_types:
                self._update_atom_type_interactions(inter_key, atom_type)
        else:
            try:
                atom_type = self.atom_type_interactions[inter_key]
            except KeyError:
                # Get lowest missing interger in self.atom_type_interactions
                atom_type = next(ifilterfalse(set(
                    self.atom_type_interactions.values()).__contains__,
                                              count(1)))
                self._update_atom_type_interactions(inter_key, atom_type)
            atom.atom_type = atom_type
        self._atom_types[atom_type].append(atom)


    def _update_atom_type_interactions(self, key, atom_type):

        """
        Adds a new key:atom_type to atom_type_interactions, if the key does not
        already exist

        Parameters
        ----------
        key : tuple
            (element, *interactions), where element is a string specifying the
            atomic element, and *interactions is one or more Interaction objects
        atom_type : int
            The atom type

        Raises
        ------
        TypeError
            If an assignment is made to an atom_type which already has
            associated interactions
        """

        if key not in self.atom_type_interactions:
            self.atom_type_interactions[key] = atom_type
        else:
            raise TypeError('assignments cannot be made to'
                            ' atom_type_interactions keys which already possess'
                            ' values')

    def add_structural_unit(self, structural_unit, force_field=None):

        """
        Adds a single structural unit to the universe, with optional force field
        applying only to that structural unit

        Parameters
        ----------
        structural_unit : StructuralUnit
            The structural unit to be added to the Universe
        force_field : str, optional
            The force field to be applied to the structural unit
        """

        structural_unit.universe = self
        self.configuration.add_structural_unit(structural_unit)
        for atom in structural_unit.atom_list:
            self.add_bonded_interaction_pairs(*atom.bonded_interaction_pairs)
            self.add_nonbonded_interaction(*atom.nonbonded_interactions)
            self._update_atom_types(atom)

        if force_field:
            self.add_force_field(force_field, structural_unit.interactions)


    def delete_structural_unit(self, structural_unit):

        """
        Deletes a single structural unit from the universe.

        Parameters
        ----------
        structural_unit : StructuralUnit
            The structural unit to be deleted from the Universe.

        Raises
        ------
        NotImplementedError
            HAS NOT BEEN IMPLEMENTED
        """


    def fill(self, structural_unit, force_field=None, **settings):

        """
        A liquid-like filling of the Universe independent of existing atoms

        Adds copies of structural_unit to existing configuration until Universe
        is full.  As exclusion region is defined by the size of a bounding
        sphere, this method is most suitable for atoms or molecules with
        approximately equal dimensions.

        CURRENT APPROACH RESULTS IN NUMBER DENSITY DIFFERENT TO WHAT IS
        SPECIFIED DEPENDING ON HOW CLOSE CUBE ROOT OF N_MOLECULES IS TO AN INT.

        Parameters
        ----------
        structural_unit : StructuralUnit
            The structural unit with which to fill the Universe
        force_field : str
            Applies a force field to the Universe
        **settings
            num_density : float
                Non-negative float specifying the number density of the
                structural unit, in units of atoms Ang ^ -3
        """

        n_units_xyz = self.dims / (1. / settings.get('num_density')) ** (1 / 3.)
        n_units_xyz = n_units_xyz.astype(int)

        positions = []
        # Determine the upper and lower bounds for structural unit with its
        # position (CoM) and its bounding box
        bounds = structural_unit.bounding_box
        mn = np.array((0., 0., 0.)) - (bounds.min - structural_unit.position)
        mx = self.dims - (bounds.min - structural_unit.position)
        for i in range(len(self.dims)):
            positions.append(np.linspace(mn[i], mx[i], n_units_xyz[i],
                                         endpoint=False))

        positions = sorted(list(product(*positions)))

        # Add the first structural unit and force field (if specified) before
        # copying the structural unit to fill the universe
        for position in positions:
            if position is positions[0]:
                self.add_structural_unit(structural_unit, force_field)
                structural_unit.position = position
            else:
                new_unit = structural_unit.copy(position)
                self.add_structural_unit(new_unit)

    def add_force_field(self, force_field, *interactions):

        """
        Adds a force field to *interactions.  If no interactions are
        passed, the force field is applied to all interactions in the Universe.

        Parameters
        ----------
        force_field : str
            The ForceField to parameterize *interactions (if provided), or all
            the interactions in the universe
        *interactions
            Interactions to parameterize with the force field
        """

        self.force_fields = force_field

        if not interactions:
            self.force_fields.parameterize_interactions(self.interactions)
        else:
            self.force_fields.parameterize_interactions(*interactions)

    def add_bonded_interaction_pairs(self, *bonded_interaction_pairs):

        """
        Adds one or more interaction pairs to the Universe

        Parameters
        ----------
        *bonded_interaction_pairs
            one or more (interaction, atoms) pairs, where atoms is a tuple of
            all atoms for that specific bonded interaction
        """

        self._bonded_interaction_pairs.update(bonded_interaction_pairs)

    def add_nonbonded_interaction(self, *nonbonded_interactions):

        """
        Adds one or more nonbonded interactions to the Universe

        Parameters
        ----------
        *nonbonded_interactions
            Nonbonded interactions to be added to the Universe
        """

        self._nonbonded_interactions.update(nonbonded_interactions)

    def solvate(self, density, solvent=None, model='SPCE', **settings):

        """
        Fills the universe with randomly distributed solvent molecules.

        Parameters
        ----------
        density : float
            The desired total density of the solvent and solute molecules,
            in units of amu Ang ^ -3
        solvent : StructuralUnit or dict
            A StructuralUnit for the solvent Molecule/Atom to fill the
            Universe with.
            A configuration dict with the position of 1 or more solvent
            StructuralUnits.
            If None then use SPCE water configuration.
        **settings
            tolerance : float
                The +/- percentage tolerance of the density to be achieved

        Raises
        ------
        NotImplementedError
            The solvate method cannot yet deal with the case where solvent
            is passed as an argument.
        """

        if settings.get('tolerance'):
            tolerance = settings['tolerance']
        if solvent is not None:
            raise NotImplementedError
            if isinstance(solvent, StructuralUnit):
                # Deal with the case where user specifies the solvent molecule
                pass
        else:
            if model == 'SPCE':
                config = copy.deepcopy(SPCE)
            else:
                # user specifies model for a different solvent.
                pass
            # Find the minimum position to set as the origin when translating
            origin = np.zeros(3)
            for mol in config['molecules'].values():
                for pos in mol.values():
                    origin = np.minimum(origin, pos)
            # Retrieve total mass of solutes in the universe.
            tot_solute_mass = 0
            for structure in self.structure_list:
                tot_solute_mass += structure.mass
            # Retrieve mass of single solvent molecule.
            solv_mass = molec_from_dict(config['molecules'].values()[0]).mass
            req_vol = solv_mass * len(config['molecules']) / density
            # Get the box dimensions required to achieve correct density
            # AT THE MOMENT ASSUMES BOX OF SOLVENT CONFIG PASSED IS CUBIC
            box_dims = np.array([req_vol ** (1. / 3)] * 3)

            difference = 0.0
            count = 0
            while True:
                count += 1
                box_dims *=  1 + difference
                print 'box dims = ', box_dims
                # Calculate the number of tiles required in each direction
                num_tiles = np.ceil(self.dims / box_dims).astype(int)
                print 'num of tiles = ', num_tiles

                mols = np.array([])
                for trans_vect in product(range(0, num_tiles[0]),
                                          range(0, num_tiles[1]),
                                          range(0, num_tiles[2])):

                    trans_tile = copy.deepcopy(config['molecules'])
                    for mol_key in trans_tile.keys():
                        mol = trans_tile[mol_key]
                        remove = False
                        for pos in mol.values():
                            if (np.linalg.norm(num_tiles) == 1
                                and difference >= 0):
                                pos = (pos - origin) * box_dims * trans_vect
                            else:
                                pos += box_dims * trans_vect - origin
                            # Check for atoms that fall outside the universe
                            if (any(pos > self.dims) or any(pos < [0, 0, 0])):
                                remove = True
                            # Check for overlap with solute molecules
                            for solute in self.molecule_list:
                                cond1 = any(pos > solute.bounding_box.min)
                                cond2 = any(pos < solute.bounding_box.max)
                                if cond1 and cond2:
                                    remove = True
                        if remove:
                            del trans_tile[mol_key]
                    mols = np.append(mols, molec_list_from_config(trans_tile))
                # Check the density
                print 'num of mols = ', len(mols)
                # print 'num of mols = ', len(trans_configs)
                actual = (len(mols) * solv_mass + tot_solute_mass) / self.volume
                print 'actual = ', actual
                difference = (actual - density) / density
                print 'diff = ', difference
                if abs(difference * 100) > abs(tolerance):
                    difference /= count
                else:
                    break
            # Once the correct density is achieved, add molecules to universe
            [self.add_structural_unit(molecule) for molecule in mols]
            print '\nIterations required: ', count


def _primitive_cubic(dimensions, number):

    """
    Generates a primitive cubic structure

    Raises
    ------
    NotImplementedError
        HAS NOT BEEN IMPLEMENTED
    """

    raise NotImplementedError


def _liquid_structure():

    """
    Generates a random arrangement of structural units

    Raises
    ------
    NotImplementedError
        HAS NOT BEEN IMPLEMENTED
    """

    raise NotImplementedError


class KSpaceSolver(object):

    """
    Class describing the k-space solver that is applied to electrostatic and/or
    dispersion interactions

    Different MD engines require different parameters to be specified for a
    k-space solver to be used. These parameters are specified in settings.

    Parameters
    ----------
    **settings
        accuracy : float
            The relative RMS error in per-atom forces

    Attributes
    ----------
    accuracy : float
        The relative RMS error in per-atom forces
    """

    def __init__(self, **settings):

        self.accuracy = settings.get('accuracy')

    @property
    def name(self):

        """
        Get the name of the class

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__


class Ewald(KSpaceSolver):

    """
    Holds the parameters that are required for the Ewald solver to be applied to
    both/either the electrostatic and/or dispersion interactions

    Parameters
    ----------
    **settings
        accuracy : float
            The relative RMS error in per-atom forces
    """

    def __init__(self, **settings):

        super(Ewald, self).__init__(**settings)


class PPPM(KSpaceSolver):

    """
    Holds the parameters that are required for the PPPM solver to be applied to
    both/either the electrostatic and/or dispersion interactions

    Parameters
    ----------
    **settings
        accuracy : float
            The relative RMS error in per-atom forces
    """

    def __init__(self, **settings):

        super(PPPM, self).__init__(**settings)

    def __eq__(self, other):

        """
        Two KSpaceSolvers are equal if their __dict__ are equal
        """

        if not isinstance(other, self.__class__):
            return False
        for k, v in self.__dict__.items():
            if v != getattr(other, k):
                return False
        return True

    def __ne__(self, other):

        return not self.__eq__(other)


class ConstraintAlgorithm(object):

    """
    Class describing the algorithm and parameters which are applied to constrain
    bonded interactions

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions

    Attributes
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    """

    def __init__(self, accuracy, max_iterations):

        self.accuracy = accuracy
        self.max_iterations = max_iterations

    @property
    def name(self):

        """
        Get the name of the class

        Returns
        -------
        str
            The name of the class
        """

        return self.__class__.__name__

    @property
    def max_iterations(self):

        """
        Get or set the maximum number of iterations that can be used when
        calculating the additional force that is required to constrain the atoms
        to satisfy the constraints on the bonded interactions

        Returns
        -------
        int
            The maximum number of iterations
        """

        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):

        self._max_iterations = int(value)


class Shake(ConstraintAlgorithm):

    """
    Holds the parameters which are required for the SHAKE algorithm to be
    applied to the constrained interactions

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions
    """

    def __init__(self, accuracy, max_iterations):

        super(Shake, self).__init__(accuracy, max_iterations)


class Rattle(ConstraintAlgorithm):

    """
    Holds the parameters which are required for the RATTLE algorithm to be
    applied to the constrained interactions

    Parameters
    ----------
    accuracy : float
        The accuracy (tolerance) of the applied constraints
    max_iterations : int
        The maximum number of iterations that can be used when calculating the
        additional force that is required to constrain the atoms to satisfy the
        constraints on the bonded interactions
    """

    def __init__(self, accuracy, max_iterations):

        super(Rattle, self).__init__(accuracy, max_iterations)


class EnergyMinimizer(object):

    """
    The MD energy minimizer

    Attributes
    ----------
    n_steps : int
        number of steps
    sz_steps : float
        size of steps
    algorithm : str
        minimization algorithm
    stop : float
        condition for minimization to stop

    Raises
    ------
    NotImplementedError
        THIS IS NOT IMPLEMENTED
    """

    def __init__(self):

        raise NotImplementedError

class Simulation(object):

    """
    Molecular dynamics engine for any ensemble

    Ensemble is defined by whether a thermostat or barostat are present

    Parameters
    ----------
    universe : Universe
        The Universe on which the simulation is performed.
    engine : str
        The MDEngine library used for the simulation.
    **settings
        temperature : float
            Simulation temperature in K.
        time_step : float
            Simulation timestep in fs.
        integrator : str
            Simulation time integrator.
        lj_options : dict
            option:value pairs for Lennard-Jones interactions.
        es_options : dict
            option:value pairs for electrostatic interactions.
        thermostat : str
            The name of the thermostat e.g. Nose-Hoover.
        barostat : str
            The name of the barostat e.g. Nose-Hoover.
        pressure : float
            Simulation pressure in Pa. This is required if a barostat is passed.

    Attributes
    ----------
    universe : Universe
        The Universe on which the simulation is performed.
    engine : MDEngine, optional
        A subclass of MDEngine which provides the interface to the MD library.
    settings : dict
        The settings passed to the Simulation.  See the Parameters section for
        details.
    """

    # TODO: Potentially separate out universe and simulation setup
    def __init__(self, universe, engine="mmtk", **settings):

        self.universe = universe
        self.settings = settings
        self.engine = MDEngineFacadeFactory.create_facade(engine)
        self._setup()

    def _setup(self):

        """
        Creates a universe within the MD engine with the equivalent
        configuration and topology to self.universe and defines the simulation
        conditions
        """

        self.engine.setup_universe(self.universe, **self.settings)
        self.engine.setup_simulation(**self.settings)

    def minimize(self, n_steps, **settings):

        """
        Minimizes the MD simulation energy

        Parameters
        ----------
        n_steps : int
            Maximum number of steps to run the minimization
        """

        self.engine.minimize(n_steps, **settings)

    def run(self, n_steps, equilibration=False):

        """
        Runs the MD simulation

        Parameters
        ----------
        n_steps : int
            Number of simulation steps to run
        equilibration : bool, optional
            If the run is for equilibration (True) or production (False).
            Default is False.
        """

        self.engine.run(n_steps, equilibration)

    @property
    def trajectory(self):

        """
        The Trajectory produced by the most recent production run of the
        Simulation.

        Returns
        -------
        Trajectory
            Most recent production run Trajectory, or None if no production run
            has been performed
        """

        try:
            return self.engine.convert_trajectory()
        except AttributeError:
            return None

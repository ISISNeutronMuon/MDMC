"""Facade for LAMMPS MD engine

This is a facade to PyLammps (added in 30th-Jul-2016 version), a convenience
wrapper for the LAMMPS Python interface i.e. where Python is extended with
LAMMPS.

Defining all interaction types requires that LAMMPS was built with the MOLECULE
package.

Note: When variables are either passed to or from PyLammps, the ctypes
conversion can mean that they are unnecessarily cast, particularly from float to
int.  This can cause issues as LAMMPS requires certain variables, e.g. number of
steps, to be int.  Therefore it is always a good idea to be cast these variables
when they are read from PyLammps e.g. int(lmp.variables['steps'].value).

Note: A minor bug in LAMMPS (Dec 2018 version) means that nangletypes returned
by PyLammps is incorrectly set to ndihedraltypes

AUTHOR :    Thomas Farmer        START DATE :    11/01/2019, 13:45:29"""


from collections import defaultdict
from copy import copy
from itertools import chain, count, product, tee
from random import randint
from tempfile import NamedTemporaryFile
import warnings

from lammps import PyLammps
import numpy as np

from MDMC.common import units
from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.MD.structural_units import Atom, BondedInteraction
from MDMC.trajectory_analysis.trajectory import TemporalConfiguration, \
    Trajectory


class LAMMPSEngine(MDEngine):

    """
    Facade for LAMMPS

    One notable issue with creating the LAMMPS facade is that some pair_styles
    are combinations of both dispersion and coulombic interactions. For example,
    although LAMMPS has lj/cut, coul/cut, and coul/long, there is no lj/long
    pair_style; it only exists in combination with coul/long
    (i.e. lj/long/coul/long). This means that the facade has to not just parse
    each nonbonded interaction, but also has to determine if that pair_style
    should be combined with another pair_style (e.g. lj/long and coul/long have
    to be combined before passing to pair_style and pair_coeff). An additional
    complication of this is that when setting the coefficients (parameters) of
    these interactions, they again have to be set together. So in theory a
    coulombic interaction needs to know the dispersion interaction it has been
    paired with an also pass those coefficients when calling pair_coeff. In
    practice a simplification can be made, at least in the case of currently
    implemented pair styles: As coulombic pair styles do no take any
    coefficients, any coulombic pair styles that comprise a combined pair_style
    can be ignored for the purposes of setting the coulombic pair_style; this
    can be taken care of when pair_coeff is called from the correspoding
    dispersion interaction, as this possesses the dispersion coefficients that
    also need to be passed when the coefficients of this pair style are set.

    Attributes:
    saved_config - the configuration from the start of the run
    time_step - a float specifying the simulation time step in fs
    temperature - a float specifying the simulation temperature in K
    skin - a float specifying the skin distance in Ang. This distance plus the
    force cutoff distance determines which atom pairs are stored in the neighbor
    list.
    neighbor_steps - an integer specifying how many steps between neighbor list
    updates
    lin_momentum_steps - an int specifying how many steps between the linear
    momentum being removed
    ang_momentum_steps - an int specifying how many steps between the angular
    momentum being removed
    pressure - a float specifying the pressure in atm
    t_damp - an int specifying over how many steps the temperature is relaxed by
    the thermostat. This only applies to Nose-Hoover, Berendsen, Langevin
    thermostats.
    p_damp - an int specifying over how many steps the pressure is relaxed by
    the barostat. This applies to all barostats.
    t_fraction - a float between 0.0 and 1.0 specifying the magnitude rescale to
    the target temperature, where 1.0 is rescale exactly to the target. This
    only applies to rescale thermostat.
    t_window - a float specifying the temperature window in K. If the
    temperature varies from the target by greater than this value, the
    temperature is rescaled. This only applies to rescale thermostats.
    thermostat - a string specifying the thermostat
    barostat - a string specifying the barostat
    atom_dict - a dictionary with {MDMC_atom: LAMMPS_atom}, where MDMC_atom is
    an MDMC Atom object and LAMMPS_atom is the corresponding LAMMPS Atom object
    atom_types - a dictionary with {type_ID: MDMC_atom_group}, where the type_ID
    is a unique integer and MDMC_atom_group is a list of atoms which are
    identical in terms of element and interactions
    system_state - a System object from the LAMMPS Python interface which
    contains properties of the simulation box
    fixes - a list of dictionaries specifying which LAMMPS fixes which are
    applied
    fix_styles - a list of strings specifying the styles of the LAMMPS fixes
    which are applied
    fix_names - a list of string specifying the names of the LAMMPS fixes which
    are applied
    """

    @property
    def saved_config(self):

        return self._saved_config

    @property
    def time_step(self):

        return self._time_step

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value):

        self._time_step = value
        try:
            # Set the timestep in LAMMPS wrapper
            self.lmp.timestep(convert_unit(self._time_step,
                                           self._time_step.unit))
        except AttributeError:
            pass

    @property
    def temperature(self):

        return self._temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value):

        self._temperature = value
        try:
            # Set the initial temperature in the LAMMPS wrapper
            self.lmp.velocity('all', 'create',
                              convert_unit(self._temperature,
                                           self._temperature.unit),
                              randint(1, 9999))
        except AttributeError:
            pass

    @property
    def skin(self):

        return self._skin

    @skin.setter
    @unit_decorator(unit=units.LENGTH)
    def skin(self, value):

        self._skin = value
        try:
            # Set the neighor list parameters in the LAMMPS wrapper
            self.lmp.neighbor(convert_unit(self._skin, self._skin.unit), 'bin')
        except AttributeError:
            pass

    @property
    def neighbor_steps(self):

        return self._neighbor_steps

    @neighbor_steps.setter
    def neighbor_steps(self, value):

        self._neighbor_steps = value
        try:
            # Set the modifiers to the neighbor list in the LAMMPS wrapper
            self.lmp.neigh_modify('every', int(self.neighbor_steps), 'delay', 0,
                                  'check', 'yes')
        except TypeError:
            pass

    @property
    def nonbonded_mix(self):

        return self._nonbonded_mix

    @nonbonded_mix.setter
    def nonbonded_mix(self, value):

        if not value:
            self._nonbonded_mix = value
        else:
            supported = ['geometric', 'arithmetic', 'sixthpower']
            if value.lower() not in supported:
                raise ValueError('The supported mixes are: {0}, {1}, {2}'
                                 ''.format(*supported))
            self._nonbonded_mix = value.lower()

    @property
    def lin_momentum_steps(self):

        return self._lin_momentum_steps

    @lin_momentum_steps.setter
    def lin_momentum_steps(self, value):

        self._lin_momentum_steps = value
        # Check for ang_momentum_steps as this is required for
        # _set_momentum_removers
        if not hasattr(self, '_ang_momentum_steps'):
            self._ang_momentum_steps = None
        # Set the momentum removers in the LAMMPS wrapper
        self._set_momentum_removers()

    @property
    def ang_momentum_steps(self):

        return self._ang_momentum_steps

    @ang_momentum_steps.setter
    def ang_momentum_steps(self, value):

        self._ang_momentum_steps = value
        # Check for lin_momentum_steps as this is required for
        # _set_momentum_removers
        if not hasattr(self, '_lin_momentum_steps'):
            self._lin_momentum_steps = None
        # Set the momentum removers in the LAMMPS wrapper
        self._set_momentum_removers()

    @property
    def pressure(self):

        return self._pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value):

        self._pressure = value

    # Unit has to be applied to getter due to operation in setter
    @property
    @unit_decorator_getter(unit=units.TIME)
    def t_damp(self):

        return self._t_damp

    @t_damp.setter
    def t_damp(self, value):

        try:
            # LAMMPS requires t_damp to be given in units of time, but it is
            # more natural to give it in units of steps - convert between them
            # here
            self._t_damp = value * self.time_step
        except TypeError:
            if value is None:
                self._t_damp = value
            else:
                raise AttributeError('the time_step attribute must be set'
                                     ' before t_damp')

    # Unit has to be applied to getter due to operation in setter
    @property
    @unit_decorator_getter(unit=units.TIME)
    def p_damp(self):

        return self._p_damp

    @p_damp.setter
    def p_damp(self, value):

        try:
            # LAMMPS requires p_damp to be given in units of time, but it is
            # more natural to give it in units of steps - convert between them
            # here
            self._p_damp = value * self.time_step
        except TypeError:
            if value is None:
                self._p_damp = value
            else:
                raise AttributeError('the time_step attribute must be set'
                                     ' before p_damp')

    @property
    def t_fraction(self):

        return self._t_fraction

    @t_fraction.setter
    def t_fraction(self, value):

        # Must be a fraction
        if value is None or 0. <= value <= 1.0:
            self._t_fraction = value
        else:
            raise ValueError('the t_fraction must be between 0.0 and 1.0')

    @property
    def t_window(self):

        return self._t_window

    @t_window.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def t_window(self, value):

        self._t_window = value

    @property
    def thermostat(self):

        return self._thermostat

    @thermostat.setter
    def thermostat(self, value):

        self._thermostat = value
        # Check for _barostat as this is required for _apply_ensemble
        if not hasattr(self, '_barostat'):
            self._barostat = None
        # Set the thermostat and barostat in LAMMPS wrapper
        self._apply_ensemble()

    @property
    def barostat(self):

        return self._barostat

    @barostat.setter
    def barostat(self, value):

        self._barostat = value
        # Check for _thermostat as this is required for _apply_ensemble
        if not hasattr(self, '_thermostat'):
            self._thermostat = None
        # Set the thermostat and barostat in LAMMPS wrapper
        self._apply_ensemble()

    @property
    def system_state(self):

        return self.lmp.system

    @property
    def fixes(self):

        return self.lmp.fixes

    @property
    def fix_styles(self):

        return [fix['style'] for fix in self.fixes]

    @property
    def fix_names(self):

        return [fix['name'] for fix in self.fixes]


    def setup_universe(self, universe, **settings):

        """
        Creates the simulation box, the atomic configuration, and the topology
        in LAMMPS

        Arguments:
        universe - an MDMC Universe object

        Settings:
        atom_style - a LAMMPS atom_style string. The default setting of 'real'
        will generally be appropriate.
        """

        self._init_lammps(**settings)
        # self.universe is set in _init_attributes
        self._init_attributes(universe)
        self._define_simulation_box(self.universe)
        self._build_configuration(self.universe)
        self._add_topology(self.universe, **settings)
        self.update_parameters()

    def setup_simulation(self, **settings):

        """
        Sets simulation parameters in LAMMPS, such as the thermodynamic
        variables, thermostat/barostat parameters and trajectory settings
        """

        self.temperature = settings.get('temperature', 300)
        self.time_step = settings.get('time_step', 1.0)
        self.traj_step = settings['traj_step']

        self._saved_config = None

        self.skin = settings.get('skin', 2.0)
        self.neighbor_steps = settings.get('neighbor_steps', 1)

        self.lin_momentum_steps = settings.get('remove_linear_momentum', 1)
        self.ang_momentum_steps = settings.get('remove_angular_momentum')

        if self.universe.constraint_algorithm:
            self._apply_constraints()

        self.pressure = settings.get('pressure')
        self.t_damp = settings.get('t_damp', 100)
        self.p_damp = settings.get('p_damp', 1000)
        self.t_window = settings.get('t_window')
        self.t_fraction = settings.get('t_fraction')
        self.rescale_step = settings.get('rescale_step')
        self.thermostat = settings.get('thermostat')
        self.barostat = settings.get('barostat')

    def minimize(self, n_steps, **settings):

        # Check fix styles for shake or rattle styles and remove them
        if 'constrain' in self.fix_names:
            self.lmp.unfix('constrain')

        self.lmp.minimize(settings.get('etol', 1.e-4),
                          settings.get('ftol', 0.),
                          n_steps,
                          settings.get('maxeval', 10000))

        # Reapply the constraints
        if self.universe.constraint_algorithm:
            self._apply_constraints()


    def run(self, n_steps, equilibration=False):

        # Remove previous dumps if they exist
        if 'traj1' in [dump['name'] for dump in self.lmp.dumps]:
            self.lmp.undump('traj1')
        # Store the trajectory in a NamedTemporaryFile
        self.trajectory_file = NamedTemporaryFile()
        # Custom trajectory output just saves the atom ID, type and positions
        self.lmp.dump('traj1', 'all', 'custom', self.traj_step,
                      self.trajectory_file.name, 'id', 'type', 'x', 'y', 'z')
        self.lmp.run(n_steps)

    def convert_trajectory(self):

        return convert_trajectory(self.trajectory_file,
                                  self.atom_type_properties,
                                  self.universe)

    def update_parameters(self):

        self._update_charges()
        self._update_bonds(self.bonds)
        self._update_angles(self.angles)
        self._update_dispersions(self.disps)

    def save_config(self):

        # It is not possible to deepcopy the LAMMPS wrapper atoms attribute, or
        # the individual atoms, so instead this saves the x, y, z, mass and
        # charge in a NumPy array with the indexes given by the atom ID (with a
        # -1 offset due to zero index)
        # The atoms attribute also is not iterable
        n_atoms = self.system_state.natoms
        atoms = np.zeros([n_atoms, 5])
        for i in range(n_atoms):
            atom = self.lmp.atoms[i]
            atoms[atom.id-1, :] = ([component for component in atom.position]
                                   + [atom.mass, atom.charge])
        self._saved_config = atoms

    def reset_config(self):

        # Raise an IndexError if the number of atoms has changed
        n_atoms = self.system_state.natoms
        if len(self.saved_config) != n_atoms:
            raise IndexError('the number of atoms has changed during the'
                             ' simulations')

        # The LAMMPS wrapper does not allow the configuration to be updated
        # simply by setting all atoms. Instead the position of the atoms must be
        # reset.
        index_components = list(enumerate(['x', 'y', 'z']))
        for id_offset in range(n_atoms):
            for index, component in index_components:
                self.lmp.set('atom', id_offset+1, component,
                             self.saved_config[id_offset][index])

    def _init_attributes(self, universe):

        """
        Initializes all attributes/properties of the LAMMPSEngine

        This partly takes the place of __init__ for LAMMPSEngine

        Arguments:
        universe - an MDMC Universe object
        """

        # Universe setup attributes
        self.universe = universe
        self.atom_dict = {}
        self.atom_types = {}
        self.atom_type_properties = []

        self.bonds = []
        self.angles = []
        self.couls = []
        self.disps = []
        self.bond_ID = {}
        self.angle_ID = {}
        self.nonbonded_mix = None

        # Simulation setup attributes
        self.time_step = None
        self.traj_step = None

        self._saved_config = None

        self.skin = None
        self.neighbor_steps = None

        self.lin_momentum_steps = None
        self.ang_momentum_steps = None

        self.temperature = None
        self.pressure = None
        self.t_damp = None
        self.p_damp = None
        self.t_window = None
        self.t_fraction = None
        self.rescale_step = None

    def _init_lammps(self, **settings):

        """
        Creates a PyLammps wrapper and sets the system of units and atom style

        Settings:
        atom_style - a string specifying the LAMMPS atom_style, which determines
        the properties that can be associated which the atoms (e.g. charge,
        bonds)
        """

        self.lmp = PyLammps()
        self.lmp.units('real')
        self.lmp.atom_style(settings.get('atom_style', 'full'))

    def _define_simulation_box(self, universe):

        """
        Defines a region and creates a simulation box that fills this region

        Arguments:
        universe - a Universe object
        """

        xlo = ylo = zlo = 0.
        xhi, yhi, zhi = universe.dims
        region_ID = 'universe'
        self.lmp.region(region_ID, 'block', xlo, xhi, ylo, yhi, zlo, zhi,
                        units='box')
        n_elements = len(universe.element_dict)

        # Determine number of bond and angle types
        bonded_interaction_types = [i.name for i in set(universe.interactions)
                                    if issubclass(type(i), BondedInteraction)]
        n_bond_types = bonded_interaction_types.count('Bond')
        n_angle_types = bonded_interaction_types.count('BondAngle')
        n_dihedral_types = 0
        n_improper_types = 0

        # Determine max number of bonds and angles per atom
        atoms = universe.atom_list
        max_bonds_per_atom = self._max_n_interaction(atoms, 'Bond')
        max_angles_per_atom = self._max_n_interaction(atoms, 'BondAngle')
        max_dihedrals_per_atom = 0
        max_improper_per_atom = 0
        self.lmp.create_box(n_elements,
                            region_ID,
                            'bond/types', n_bond_types,
                            'angle/types', n_angle_types,
                            'dihedral/types', n_dihedral_types,
                            'improper/types', n_improper_types,
                            'extra/bond/per/atom', max_bonds_per_atom,
                            'extra/angle/per/atom', max_angles_per_atom,
                            'extra/dihedral/per/atom', max_dihedrals_per_atom,
                            'extra/improper/per/atom', max_improper_per_atom
                           )

    def _build_configuration(self, universe):

        """
        Adds atoms to LAMMPS

        Arguments:
        universe - a Universe object
        """

        self.atom_types = universe.atom_types
        # Assume all atoms of the same type have the same element and mass
        # Sort atoms based on atom_type (i.e. numerically starting at 1) so
        # that it is possible to index into atom_type_properties with
        # atom_type - 1 (where the -1 is to account for 0 indexing on lists)
        self.atom_type_properties = [(atom[0].element, atom[0].mass)
                                     for type, atom
                                     in sorted(self.atom_types.items())]

        for type_ID, atom_type_group in self.atom_types.items():
            # The mass below has associated units, which causes a segfault when
            # it is passed to LAMMPS - cast to float to remove units
            self.lmp.mass(type_ID, float(atom_type_group[0].mass))
            for atom in atom_type_group:
                self.lmp.create_atoms(type_ID, 'single', *atom.position)
                self.atom_dict[atom] = self.lmp.atoms[self.lmp.atoms.natoms - 1]

    def _max_n_interaction(self, atoms, name):

        """
        Arguments:
        atoms - a list of Atom objects
        name - a string specifying an Interaction type, for example 'Bond'

        Returns:
        int specifying the maximum number of interactions with a given name that
        any atom possesses
        """

        return max([len(filter(lambda i: i.name == name, atom.interactions))
                    for atom in atoms])

    def _add_topology(self, universe, **settings):

        bonds, angles, disps, couls, others = partition_interactions(
            set(universe.interactions),
            ['Bond', 'BondAngle', 'Dispersion', 'Coulombic'],
            unpartitioned=True,
            lst=True)

        if others:
            raise NotImplementedError('This interaction type has not been'
                                      ' implemented in the LAMMPS facade')

        self.bonds = bonds
        self.angles = angles
        self.disps = disps
        self.couls = couls

        # LAMMPS uses pair_style for all nonbonded interactions, so dispersive
        # and coulombic interactions are treated together. While multiple
        # identical pair_styles can be used with the hybrid command, it is
        # inefficient, so duplicates are removed with set.
        nonbonded_styles = parse_all_nonbonded_styles(disps+couls)
        if nonbonded_styles:
            self._set_kspace_solver()
            self.lmp.pair_style('hybrid/overlay', *nonbonded_styles)
            self._create_coulombic(couls)
            self._update_charges()
            self.nonbonded_mix = settings.get('nonbonded_mix')
            # Dispersion creation and updating are the same, so only an update
            # method exists
            self._update_dispersions(disps)
            # Apply LAMMPS modifications to nonbonded interactions
            self._modify_nonbonded_styles(couls+disps)

        if bonds:
            # Set used to remove duplicate bond styles, which are not required
            # to be (and in fact cannot) be passed to LAMMPS bond_style
            self.lmp.bond_style('hybrid',
                                *set(tuple([parse_bonded_styles(b)
                                            for b in bonds])))
            self._create_bonds(bonds)

        if angles:
            # Set used to remove duplicate bond styles, which are not required
            # to be (and in fact cannot) be passed to LAMMPS angle_style
            self.lmp.angle_style('hybrid',
                                 *set(tuple([parse_bonded_styles(a)
                                             for a in angles])))
            self._create_angles(angles)

    def _create_coulombic(self, couls):

        """
        Creates the coulombic interactions in LAMMPS

        AS MDMC CURRENTLY ONLY CONSIDERS COULOMBIC INTERACTIONS BETWEEN
        LIKE-LIKE ATOMS, THE CROSS TERM IS INFERRED FROM THESE RATHER THAN
        PASSED EXPLICITLY - THIS CAN LEAD TO UNPREDICTABLE BEHAVIOUR IF MORE
        THAN ONE STYLE OF COULOMBIC INTERACTION IS USED.

        Arguments:
        couls - a list of coulombic interactions
        """

        # Coulombic interaction doesn't require parameter setting, as this is
        # handled by the atom property charge
        # As Coulombic interactions in MDMC only have one type, that interaction
        # style (e.g. Coulomb) is applied to the interactions between that type
        # and all other types (achieved in LAMMPS with '*' notation). As
        # interactions are overwritten, it is the style of last atom_type
        # that determines its unlike interactions.

        all_styles = parse_all_nonbonded_styles(couls+self.disps)
        for coul in couls:
            coul_style = parse_nonbonded_styles(coul)[0]
            for style in all_styles:

                # As is explained in the LAMMPSEngine docstring, pair_coeffs for
                # coulombic interactions can only be set if the pair_style is
                # not part of a combined pair_style (e.g. lj/long/coul/long).
                # Therefore below the style of each coulombic interaction must
                # exactly match one of all of the nonbonded styles in the
                # simulation, otherwise the pair coeff is not set here; it is
                # instead set by the corresponding dispersion interaction, which
                # possesses the coefficients (parameters) which also need to be
                # passed to pair_coeff.
                if isinstance(style, str) and coul_style == style:
                    for atom_type in coul.atom_types:
                        self.lmp.pair_coeff(atom_type, '*', style)

    def _update_charges(self):

        """
        Updates the charges in LAMMPS
        """

        for atom, L_atom in self.atom_dict.items():
            self.lmp.set('atom',
                         L_atom.id,
                         'charge',
                         convert_unit(atom.charge, atom.charge.unit))

    def _update_dispersions(self, disps):

        """
        Updates dispersion interactions in LAMMPS

        Arguments:
        disps - a list of dispersion interactions
        """

        all_styles = parse_all_nonbonded_styles(disps+self.couls)

        for disp in disps:
            atom_type_pairs = product(disp.atom_types[0], disp.atom_types[1])
            disp_style = parse_nonbonded_styles(disp)[0]
            for style in all_styles:
                if isinstance(style, str) and disp_style in style:
                    coeffs = parse_dispersion_coefficients(disp, disp_style)

                    # LAMMPS uses mixing rules to set coefficients for undefined
                    # unlike atom_type pairs (e.g. if LJ interactions exist for
                    # 1 1 and 2 2 atom types, but not for 1 2). For this to
                    # occur, all i=j (i.e. like) interactions must be defined -
                    # below these are all set to zero. These zero coeffs will
                    # then be overwritten with the correct values (in the
                    # atom_type_pair for loop) if they are defined in MDMC.
                    zero_coeffs = [0. for _ in coeffs]
                    if self.nonbonded_mix:
                        for atom_type in self.atom_types.keys():
                            self.lmp.pair_coeff(atom_type,
                                                atom_type,
                                                style,
                                                *zero_coeffs)
                    # If not mixing is to occur, all permutations of atom types
                    # have coefficients set to zero. Again, the zero coeffs will
                    # be overwritten with the correct values (in the
                    # atom_type_pair for loop) if they are defined in MDMC.
                    else:
                        self.lmp.pair_coeff('*',
                                            '*',
                                            style,
                                            *zero_coeffs)

                    for atom_type_pair in atom_type_pairs:
                        self.lmp.pair_coeff(atom_type_pair[0],
                                            atom_type_pair[1],
                                            style,
                                            *coeffs)

    def _modify_nonbonded_styles(self, nonbonded_interactions):

        """
        Applies modifications to nonbonded pair styles

        Arguments:
        nonbonded_interactions - a list of nonbonded interactions which will
        have modifications applied to the corresponding pair styles
        """

        # LAMMPS pair_modify is of the following form:
        # pair_modify('pair', 'lj/cut', 'mix', 'geometric', 'tail', 'yes')
        # pair_modify('coul/long', 'mix', 'arithmetic')
        # where each pair style (lj/cut, coul/long etc) has any mix or tail
        # keywords defined after the pair style. If the same pair_style occurs
        # multiple times but with different modifiers
        # (e.g. 'lj/cut', 'mix', 'geometric' and 'lj/cut', 'mix', 'arithmetic')
        # whichever pair_modify occurs last will be applied to all identical
        # pair_styles.

        # Some pair styles cannot have vdw tail corrections - exclude these and
        # warn about this exclusion
        excluded = ['lj/long/coul/long']

        # Determine all pair_styles by parsing all nonbonded styles and removing
        # the numerical values (e.g. if a cutoff is defined). Parsing all
        # nonbonded styles has the effect of combining some pair styles that
        # cannot be passed to lammps individually (e.g. lj/long/coul/long)
        all_styles = [style for style
                      in parse_all_nonbonded_styles(nonbonded_interactions)
                      if isinstance(style, str)]

        all_inter_str = []
        for inter in nonbonded_interactions:
            inter_style = parse_nonbonded_styles(inter)[0]
            # Effectively tests if the parsed pair style of an interaction
            # occurs within a combined pair style (e.g. lj/long occurs within
            # the combined style lj/long/coul/long) - the combined style is then
            # used, as this is what will be recognised by LAMMPS
            for style in all_styles:
                if inter_style in style:
                    inter_str = (style, )
                    if self.nonbonded_mix:
                        inter_str += ('mix', self.nonbonded_mix)
                    # try/except to account for nonbonded interaction types
                    # which do not have the modification type defined
                    try:
                        # Applies the vdw tail correction to the energy and
                        # pressure
                        if inter.vdw_tail_correction:
                            # warn if user attempts to apply vdw tail correction
                            # to a pair style that is not allowed by LAMMPS
                            if style in excluded:
                                warnings.warn('{0} pair style cannot have a'
                                              ' vdw tail correction'
                                              ' applied'.format(style))
                            else:
                                inter_str += ('tail', 'yes')
                    except AttributeError:
                        pass
                    # If either a tail or a mix has been added to the inter_str
                    # add this to the list of all_inter_str
                    if inter_str != (style, ):
                        all_inter_str.append(inter_str)
        if all_inter_str:
            # if there are multiple identical interaction types then the
            # inter_str of each will be duplicated in all_inter_str e.g.
            # inter_str == [('lj/cut', 'tail', 'yes'),
            #               ('lj/cut', 'tail', 'yes')]
            # Remove these duplicates with set
            #
            # It should then be possible (based on LAMMPS documentation) to pass
            # this set (with tuples unpacked) at once to pair_modify - however
            # LAMMPS has a bug preventing this. Instead pass these tuples
            # individually. LAMMPS does not overwrite the affects of previous
            # pair_modify commands, as long as they are applied to different
            # pair_styles.
            lmp_str = set(all_inter_str)
            for pair_style in lmp_str:
                self.lmp.pair_modify('pair', *pair_style)

    def _create_bonds(self, bonds):

        """
        Creates coefficients and bonds in LAMMPS, and fills the bond_ID
        dictionary with bond: ID pairs

        Arguments:
        bonds - a list of bond interactions
        """

        special = 'no'
        for ID, bond in enumerate(bonds, start=1):
            # Create the bond coefficients
            self.lmp.bond_coeff(ID, *parse_bonded_coefficients(bond))

            # Relate each bond with its ID
            self.bond_ID[bond] = ID

            # Create the bonds
            # Special triggers the internal interaction list in LAMMPS
            # This must at least occur at the end, and is an expensive
            # operation
            if bond is bonds[-1]:
                special = 'yes'
            for atom_tpl in bond.atoms:
                atom_IDs = [self.atom_dict[atom].id for atom in atom_tpl]
                self.lmp.create_bonds('single/bond',
                                      ID,
                                      atom_IDs[0],
                                      atom_IDs[1],
                                      'special',
                                      special)

    def _update_bonds(self, bonds):

        """
        Updates the bond coefficients, which are then applied to any bonds which
        have previously been set

        Arguments:
        bonds - a list of bond interactions
        """

        for bond in bonds:
            self.lmp.bond_coeff(self.bond_ID[bond],
                                *parse_bonded_coefficients(bond))

    def _create_angles(self, angles):

        """
        Creates coefficients and angles in LAMMPS, and fills the angle_ID
        dictionary with angle: ID pairs

        Arguments:
        angles - a list of bond angle interactions
        """

        special = 'no'
        for ID, angle in enumerate(angles, start=1):
            # Create the bond coefficients
            self.lmp.angle_coeff(ID, *parse_bonded_coefficients(angle))

            # Relate each bond with its ID
            self.angle_ID[angle] = ID

            # Create the angles
            # Special triggers the internal interaction list in LAMMPS
            # This must at least occur at the end, and is an expensive
            # operation
            if angle is angles[-1]:
                special = 'yes'
            for atom_tpl in angle.atoms:
                atom_IDs = [self.atom_dict[atom].id for atom in atom_tpl]
                # angles are also created with lmp.create_bonds, just with a
                # keyword of single/angle
                self.lmp.create_bonds('single/angle',
                                      ID,
                                      atom_IDs[0],
                                      atom_IDs[1],
                                      atom_IDs[2],
                                      'special',
                                      special)

    def _update_angles(self, angles):

        """
        Updates the angle coefficients, which are then applied to any angles
        which have been previously set

        Arguments:
        angles - a list of bond angle interactions
        """

        for angle in angles:
            self.lmp.angle_coeff(self.angle_ID[angle],
                                 *parse_bonded_coefficients(angle))

    def _set_kspace_solver(self):

        """
        Creates a k-space solve in LAMMPS using kspace_style, if one is required

        Uses either the kspace_solver, the electrostatic_solver or both the
        electrostatic_solver and dispersive_solver attribute of the MDMC
        universe to set the kspace_style. Note that LAMMPS does not support
        different electrostatic and dispersive solvers. Setting with equivalent
        electrostatic and dispserive solvers is equivalent to setting with
        kspace_solver. LAMMPS also does not support just applying a dispersive
        solver.
        """

        kspace = self.universe.kspace_solver
        electrostatic = self.universe.electrostatic_solver
        dispersive = self.universe.dispersive_solver


        # LAMMPS supports a single kspace solver, which can be taken from kspace
        # electrostatic or dispersive solvers of the universe. If any other
        # solver is defined as well as kspace (which effecitvely defines both
        # other solves), raise an error.
        if kspace and (electrostatic or dispersive):
            raise TypeError('kspace solver cannot be applied in conjunction'
                            ' with either electrostatic or dispersive solvers')
        if kspace:
            self.lmp.kspace_style(*parse_kspace_solver(kspace))
        # If either an electrostatic or a dispersive solver has been defined,
        # use this. If both have been defined, this is valid as long as they are
        # equivalent.  If not, raise an error.
        if electrostatic:
            if dispersive and electrostatic != dispersive:
                raise TypeError('LAMMPS only supports a single solver, so if'
                                ' both dispersive and electrostatic solvers'
                                ' have been defined then they must be'
                                ' equivalent')
            self.lmp.kspace_style(*parse_kspace_solver(electrostatic))
        elif dispersive:
            raise TypeError('LAMMPS does not support only applying a dispersive'
                            ' solver - it must be applied in conjunction with'
                            ' an electrostatic solver')


    def _set_momentum_removers(self):

        """
        Creates the fixes in LAMMPS which remove the linear and angular momentum
        of the simulation

        Removes all pre-existing momentum remover fixes
        """

        for name in self.fix_names:
            if name in ['RemoveMomentum', 'RemoveLinearMomentum',
                        'RemoveAngularMomentum']:
                self.lmp.unfix(name)

        if self.lin_momentum_steps == self.ang_momentum_steps is not None:
            self.lmp.fix('RemoveMomentum', 'all', 'momentum',
                         self.lin_momentum_steps, 'linear', 1, 1, 1, 'angular')
        else:
            if self.lin_momentum_steps:
                self.lmp.fix('RemoveLinearMomentum', 'all', 'momentum',
                             self.lin_momentum_steps, 'linear', 1, 1, 1)
            if self.ang_momentum_steps:
                self.lmp.fix('RemoveAngularMomentum', 'all', 'momentum',
                             self.ang_momentum_steps, 'angular')

    def _apply_constraints(self):

        """
        Adds a constraint fix to LAMMPS for all bonds and bond angles which are
        constrained
        """

        # Sort bonded interactions in the Universe which are constrained into
        # bonds and angles
        b_inters = set(self.universe.bonded_interactions)
        bonds, angles = partition_interactions([inter for inter
                                                in b_inters
                                                if inter.constrained],
                                               ['Bond', 'BondAngle'], lst=True)
        algorithm = parse_constraint(self.universe.constraint_algorithm,
                                     bonds=bonds, bond_ID_dict=self.bond_ID,
                                     angles=angles, angle_ID_dict=self.angle_ID)

        # Create a group from all of the atom types in the constrained bonds and
        # angles - the fix will be applied to this group
        # chain is used to flatten inter.atoms, which is a list of tuples
        atom_types = set([atom.atom_type for inter in bonds+angles
                          for atom in chain.from_iterable(inter.atoms)])
        constrain_group = 'constrain_group'
        self.lmp.group(constrain_group, 'type', *atom_types)
        self.lmp.fix('constrain', constrain_group, *algorithm)

    def _apply_ensemble(self):

        """
        Passes the required LAMMPS fixes to apply a specific thermodynamic
        ensemble to the simulation

        Removes all pre-existing thermostat and barostat fixes
        """

        # Remove thermostat and barostat fixes
        for name in self.fix_names:
            if name in ['nve', 'nvt', 'npt', 'nph', 't_berendsen',
                        'p_berendsen', 'langevin', 'rescale']:
                self.lmp.unfix(name)

        if not self.thermostat and not self.barostat:
            self.lmp.fix('nve', 'all', 'nve')
        else:
            if self.thermostat:
                temp = convert_unit(self.temperature, self.temperature.unit)
                if not self.thermostat == 'rescale':
                    t_damp = convert_unit(self.t_damp, self.t_damp.unit)
                    thermo_params = [temp, temp, t_damp]
            if self.barostat:
                press = convert_unit(self.pressure, self.pressure.unit)
                p_damp = convert_unit(self.p_damp, self.p_damp.unit)
                press_params = ['iso', press, press, p_damp]

            # Apply thermostat
            if self.thermostat == 'nose':
                if self.barostat == 'nose':
                    self.lmp.fix('npt', 'all', 'npt', 'temp',
                                 *thermo_params + press_params)
                else:
                    self.lmp.fix('nvt', 'all', 'nvt', 'temp', *thermo_params)
            elif self.thermostat == 'berendsen':
                self.lmp.fix('t_berendsen', 'all', 'temp/berendsen',
                             *thermo_params)
            elif self.thermostat == 'langevin':
                self.lmp.fix('langevin', 'all', 'langevin',
                             *thermo_params + [randint(0, 9999)])
            elif self.thermostat == 'rescale':
                # temp/rescale does not do time integration so also requires nve
                t_window = convert_unit(self.t_window, self.t_window.unit)
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('rescale', 'all', 'temp/rescale',
                             self.rescale_step, temp, temp, t_window,
                             self.t_fraction)
            # Apply barostat
            if self.barostat == 'berendsen':
                self.lmp.fix('p_berendsen', 'all', 'press/berendsen',
                             *press_params)
            elif self.barostat == 'nose' and not self.thermostat == 'nose':
                self.lmp.fix('nph', 'all', 'nph', *press_params)


# Define the unit system used in LAMMPS
# NB: LAMMPS uses deg for angle but radian for derived quantities of angle:
# e.g. harmonic angle potential strength is in kcal / mol radian ^ 2
SYSTEM = {
    'LENGTH':units.Unit('Ang'),
    'TIME':units.Unit('fs'),
    'MASS':units.Unit('g') / units.Unit('mol'),
    'CHARGE':units.Unit('e'),
    'ANGLE':units.Unit('deg'),
    'TEMPERATURE':units.Unit('K'),
    # AMOUNT is required for unit conversion of energies to work
    'AMOUNT':units.Unit('mol'),
    'ENERGY':units.Unit('kcal') / units.Unit('mol'),
    'FORCE':units.Unit('kcal') / (units.Unit('Ang') * units.Unit('mol')),
    'PRESSURE':units.Unit('atm')
}


def convert_unit(value, unit, to_LAMMPS=True):

    """
    Converts between MDMC units and LAMMPS real units

    Arguments:
    value - a float specifying the value in MDMC units
    unit - the unit of the value
    to_LAMMPS - a boolean specifying if the conversion is from MDMC units to
    LAMMPS units

    Returns:
    a float or array with the value in LAMMPS units if to_LAMMPS is True,
    otherwise a float or array with the value in MDMC units
    """

    def expand_components(unit):

        """
        Expands out the components of a unit, so that the unit is expressed
        purely in terms of base units

        Returns:
        A tuple of (num, denom), where num is a list of all base units in the
        numerator, and denom is a list of all base units in the denominator
        """

        num, denom = [], []
        if unit.base:
            num.append(unit)
        else:
            # tuple unpacking style below appends to num and denom lists
            for comp in unit.components['numerator']:
                num[len(num):], denom[len(denom):] = expand_components(comp)
            for comp in unit.components['denominator']:
                denom[len(denom):], num[len(num):] = expand_components(comp)

        return num, denom

    # Expand the unit in terms of its base units (for numerator and denominator)
    if to_LAMMPS:
        L_SYS = copy(SYSTEM)
        # For angular potential strength LAMMPS requires the units in rad,
        # rather than degrees (which is uses otherwise). Therefore if the unit
        # is in MDMC angular potential strength units (energy / angle^2), the
        # ANGLE entry in SYSTEM is replaced by radians.
        if unit == units.SYSTEM['ENERGY'] / units.SYSTEM['ANGLE'] ** 2:
            L_SYS['ANGLE'] = units.Unit('rad')

        expanded_unit = expand_components(unit)
        SYSTEM_INV = {unit:property for property, unit in units.SYSTEM.items()}
        # Apply inversion to all components
        unit_nums, unit_denoms = map(lambda comp_list: [L_SYS[SYSTEM_INV[comp]]
                                                        for comp in comp_list],
                                     expanded_unit)
    else:
        unit_denoms, unit_nums = expand_components(unit)

    conv_nums, conv_denoms = [], []
    for component in unit_nums:
        conv_nums[len(conv_nums):], conv_denoms[len(conv_denoms):] = \
            expand_components(component)
    for component in unit_denoms:
        conv_denoms[len(conv_denoms):], conv_nums[len(conv_nums):] = \
            expand_components(component)

    for component in conv_nums:
        value *= getattr(units, component)
    for component in conv_denoms:
        value /= getattr(units, component)

    return value


def parse_bonded_styles(interaction):

    """
    Converts MDMC InteractionFunction names for BondedInteractions to LAMMPS
    bond styles

    Arguments:
    interaction - an MDMC interaction

    Returns:
    a string with the corresponding LAMMPS bond style
    """

    if interaction.function_name == 'HarmonicPotential':
        return 'harmonic'
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')


def parse_nonbonded_styles(interaction):

    """
    Converts MDMC InteractionFunction names for NonBondedInteractions to LAMMPS
    pair styles

    Arguments:
    interaction - an MDMC interaction

    Returns:
    a list with the correspoding LAMMPS pair style
    """

    lmp_str = []
    if interaction.function_name == 'LennardJones':
        lmp_str.append('lj/')
    elif interaction.function_name == 'Coulomb':
        lmp_str.append('coul/')
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    if interaction.cutoff:
        cutoff = convert_unit(interaction.cutoff, interaction.cutoff.unit)
        kspace = interaction.universe.kspace_solver
        electrostatic = interaction.universe.electrostatic_solver
        dispersive = interaction.universe.dispersive_solver
        if kspace or (electrostatic and interaction.name == 'Coulombic') \
            or (dispersive and interaction.name == 'Dispersion'):
            lmp_str[-1] += 'long'
        else:
            lmp_str[-1] += 'cut'
        lmp_str.append(cutoff)
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    return lmp_str


def parse_all_nonbonded_styles(interactions):

    """
    Converts all NonBondedInteractions to LAMMPS pair styles

    This is required because LAMMPS frequently treats Coulombic and Disperion
    interactions together; these cases need to be dealt with to generate the
    correct input to LAMMPS pair styles.  For example, while the pair_styles
    'lj/cut', 'coul/cut' and 'coul/long' can all be passed separately, 'lj/long'
    only exists as part of other pair styles, such as 'lj/long/coul/long'.

    IF A NONBONDED STYLE COULD FORM PART OF TWO PAIRS THEN THE FIRST PAIR THAT
    OCCURS WILL BE USED (ALTHOUGH THIS SCENARIO SHOULD NOT OCCUR)

    Arguments:
    interactions - A list of MDMC NonBondedInteractions

    Returns:
    a list with all of the LAMMPS pair styles
    """

    # Set to remove duplicates
    parsed_interactions = list(set([tuple(parse_nonbonded_styles(nb))
                                    for nb in interactions]))
    # Parsed interactions will be a list of tuple(style, parameters) e.g.
    # ('lj/cut', cutoff). Chain from iterable to flatten this for ease of
    # searching for pair styles
    flat_interactions = list(chain.from_iterable(parsed_interactions))

    # Check for coulombic and dispersion pairs that need to be combined
    # Dispersion styles always precede coulombic styles in LAMMPS pair styles
    disp_styles = ['lj/long']
    coul_styles = ['coul/long']

    lmp_str = []
    # Iterate over all pairs - this will need refactoring if all disp styles
    # cannot be combined with all coul styles
    # While this is a very inefficient solution, there will be so few nonbonded
    # styles it is irrelevant
    for d_style in disp_styles:
        for c_style in coul_styles:
            if c_style in flat_interactions and d_style in flat_interactions:
                # Iterate over all parsed interactions for each style
                for int1 in parsed_interactions:
                    for int2 in parsed_interactions:
                        if d_style == int1[0] and c_style == int2[0]:
                            lmp_str.append('/'.join([d_style, c_style]))
                            if lmp_str[-1] == 'lj/long/coul/long':
                                lmp_str.append('long long')
                            lmp_str.append(int1[1])
                            if int1[1] != int2[1]:
                                if lmp_str[-3] == 'lj/long/coul/long':
                                    raise ValueError('LAMMPS requires both'
                                                     ' cutoffs to be the same'
                                                     ' for long range LJ and'
                                                     ' coulombic pair styles')
                                lmp_str.append(int2[1])
                            parsed_interactions.remove(int1)
                            parsed_interactions.remove(int2)
    # Include all pair styles that were not part of a merged pair i.e.
    # everything left over in parsed_interactions
    # Chain used to flatten list of tuples
    try:
        return list(chain.from_iterable(lmp_str + parsed_interactions))
    except TypeError:
        return lmp_str + parsed_interactions


def parse_bonded_coefficients(interaction):

    """
    Orders MDMC Parameters for input to LAMMPS bond_coeff and angle_coeff

    Arguments:
    interaction - an MDMC interaction

    Returns:
    A list of style and parameters converted to the input format for LAMMPS
    bond_coeff and angle_coeff
    """

    parameters = {p.name:convert_unit(p.value, p.unit)
                  for p in interaction.params}
    style = parse_bonded_styles(interaction)

    if style == 'harmonic':
        ordered_parameters = [parameters['potential_strength'],
                              parameters['equilibrium_state']]
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    return [style] + ordered_parameters


def parse_dispersion_coefficients(interaction, style):

    """
    Orders MDMC Parameters for input to LAMMPS pair_coeff

    Arguments:
    interaction - an MDMC interaction object
    style - a LAMMPS pair_style

    Returns:
    A list of parameters converted to the input format for LAMMPS pair_coeff
    """

    parameters = {p.name:convert_unit(p.value, p.unit)
                  for p in interaction.params}

    if 'lj' in style:
        ordered_parameters = [parameters['epsilon'],
                              parameters['sigma']]
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')
    return ordered_parameters


def parse_kspace_solver(solver):

    """
    Converts an MDMC kspace solver for input to LAMMPS kspace_style

    Arguments:
    solver - an MDMC kspace solver

    Returns:
    A list of style and parameters for input to LAMMPS kspace_style
    """

    lmp_str = []

    # Add algorithm name
    if solver.name.lower() == 'ewald':
        lmp_str.append('ewald')
    elif solver.name.lower() == 'pppm':
        lmp_str.append('pppm')
    else:
        raise NotImplementedError('This k-space solver has not been implemented'
                                  ' in the LAMMPS facade')

    # Add accuracy
    lmp_str.append(solver.accuracy)

    return lmp_str

def parse_constraint(constraint_algorithm, bonds=[], bond_ID_dict={}, angles=[],
                     angle_ID_dict={}):

    """
    Converts an MDMC constraint algorithm for input to LAMMPS fix, or raises a
    NotImplementedError if the algorithm does not exist within LAMMPS

    At least one of bonds and angles must be passed

    Arguments:
    constraint_algorithm - an object which derives from ConstraintAlgorithm
    bonds - a list of constrained Bonds
    bond_ID_dict - a dictionary with bond: ID pairs where bond is a Bond object
    and ID is the integer in LAMMPS which refers to the bond
    angles - a list of constrained BondAngles
    angle_ID_dict - a dictionary with angle: ID pairs where angle is a BondAngle
    object and ID is the integer in LAMMPS which refers to the angle

    Returns:
    A list of input parameters for LAMMPS fix, not including the first two
    terms (fix ID, group-ID).  The output list is:

    [algorithm name, accuracy, max iterations, 'b', bond IDs, 'a', angle IDs]

    where the last four entries are optional, although a minimum of two is
    required.
    """

    # Raise error if there is not at least one constrained interaction passed
    if not (bonds or angles):
        raise TypeError('A LAMMPS constraint fix must have constraints on at'
                        ' least one bond or one bond angle')

    lmp_str = []

    # Add algorithm name
    if constraint_algorithm.name.upper() == 'SHAKE':
        lmp_str.append('shake')
    elif constraint_algorithm.name.upper() == 'RATTLE':
        lmp_str.append('rattle')
    else:
        raise NotImplementedError('This constraint is not implemented in the'
                                  ' LAMMPS facade')

    # Add accuracy and max iterations
    lmp_str.append(float(constraint_algorithm.accuracy))
    lmp_str.append(int(constraint_algorithm.max_iterations))

    # Never display the constraint statistics
    lmp_str.append(0)

    # Add bonds and their LAMMPS IDs and angles and their LAMMPS IDs
    if bonds:
        lmp_str.append('b')
        lmp_str += [bond_ID_dict[bond] for bond in bonds if bond.constrained]
    if angles:
        lmp_str.append('a')
        lmp_str += [angle_ID_dict[angle] for angle in angles
                    if angle.constrained]

    return lmp_str


def partition(items, predicate):

    """
    Partitions an iterable using a predicate

    Arguments:
    items - an iterable
    predicate - a predicate that can be applied to items to returned True or
    False

    Returns:
    A tuple of (gen_true, gen_false), where gen_true is a generator of all items
    for which the predicate returned True, and gen_false is a generator of all
    items for which the predicate returned False
    """

    a, b = tee((predicate(item), item) for item in items)
    return ((item for pred, item in a if pred),
            (item for pred, item in b if not pred))


def partition_interactions(interactions, names, unpartitioned=False, lst=False):

    """
    Partitions an iterable of Interaction objects using a list of Interaction
    names

    This occurs by using partition to filter out one Interaction type for each
    loop, so previously identified Interactions are no longer considered.

    Arguments:
    interactions - an iterable of Interaction objects
    names - a list of names of Interaction classes
    unpartitioned - a boolean
    lst - a boolean

    Returns:
    A tuple of length len(names) where index n is a generator of all of the
    Interaction objects which have the name specified by names[n]. For example:

    bonds, angles = partition_interactions(interactions, ['Bond, BondAngle'])

    If unpartitioned=True then a generator containing any Interaction objects
    that did not have a name in names is returned as an additional item in the
    tuple.
    If lst=True then the returned n-length tuple contains lists of all of
    the Interaction objects which have the name specified by names[n].
    """

    interaction_lst = [None] * len(names)
    i = 0
    for name in names:
        predicate = lambda x, n=name: x.name == n
        interaction_lst[i], interactions = partition(interactions, predicate)
        i += 1
    if unpartitioned:
        interaction_lst += [interactions]
    if lst:
        interaction_lst = [list(i) for i in interaction_lst]
    return tuple(interaction_lst)


def convert_trajectory(trajectory_file, atom_type_properties, universe=None,
                       start=0, stop=None, step=1, scaled_positions=False,
                       atom_IDs=None):

    """
    Converts between a LAMMPS trajectory dump and an MDMC trajectory

    The LAMMPS dump must include at least id, atom_type, and xyz positions. The
    xyz positions must be consecutive and in that order. The same is true of the
    xyz components of the velocity, if they are provided.

    Arguments:
    trajectory_file - a string specifying the LAMMPS trajectory filename
    atom_type_properties - a list of tuples (symbol, mass) for all atom_types
    (ordered) by atom_type, where symbol is a string specifying the element of
    the atom_type and mass is a float specifying the mass of the atom_type
    universe - an MDMC universe
    start - an integer specifying the first trajectory, inclusive
    start - an integer specifying the last trajectory, exclusive
    step - an integer specifying the step size between trajectories
    scaled_positions - a boolean specifying if the LAMMPS trajectory file
    provides the positions in scaled coordinates (i.e. xs, ys, yz)
    atom_IDs - a list specifying the LAMMPS IDs of the atoms which should be
    converted. If None then all atoms are converted.
    """

    def create_atom(line):
        LAMMPS_ID = line[i_id]
        atom_type = int(line[i_type])
        # If distance units are same for MDMC and LAMMPS then
        # don't call convert_units - currently hardcoded
        # Same goes for velocity and time units
        position = [float(splt) for splt in line[i_pos:i_pos+3]]
        # Get symbol and mass from atom_type_properties
        # Adjusted for 0 index
        symbol, mass = atom_type_properties[atom_type-1]
        atom = Atom(symbol, position=position, mass=mass)
        atom.atom_type = atom_type
        if universe:
            atom.universe = universe
        if i_vel is not None:
            atom.velocity = [float(splt) for splt
                             in line[i_vel:i_vel+3]]
        return atom

    # Change expected position string if scaled positions are used
    pos_string = 'xs' if scaled_positions else 'x'

    configs = []
    config_iter = start
    config_indexes = count(start, step)
    next_iter = config_indexes.next()
    with open(trajectory_file.name, 'r') as file_handler:
        line = file_handler.readline()
        while line:

            if 'ITEM: TIMESTEP' in line:
                line = file_handler.readline()
                time_step = int(line.split()[0])

            if 'ITEM: NUMBER OF ATOMS' in line:
                line = file_handler.readline()
                n_atoms = int(line.split()[0])
                # Check that n_atoms is as expected, if a universe was passed
                if universe:
                    assert n_atoms == len(universe.atom_list)

            if 'ITEM: BOX BOUNDS' in line:
                # CURRENTLY ASSUMES ORTHOGONAL SIMULATION BOX
                if 'xy' in line:
                    raise TypeError('triclinic simulation boxes have not'
                                    ' been implemented')
                # Test dimensions are as expected, if a universe was passed
                # CURRENTLY ASSUMES VOLUME IS CONSERVED
                if universe:
                    for i in range(3):
                        line = file_handler.readline()
                        min, max = [float(splt) for splt in line.split()]
                        assert min == 0.0
                        # unit is taken from array as dims is a UnitArray
                        assert max == convert_unit(universe.dims[i],
                                                   universe.dims.unit)

            if 'ITEM: ATOMS' in line:

                if config_iter == start:
                    # Determine order of LAMMPS atom properties
                    # Assumes that position components (x y z) and velocity
                    # components (vx vy vz) are always adjacent and ordered as
                    # shown
                    splt = line.split()
                    i_id, i_type, i_pos = [splt.index(prop) - 2 for prop
                                           in ['id', 'type', pos_string]]
                    if 'vx' in splt:
                        i_vel = splt.index('vx')
                    else:
                        i_vel = None

                if config_iter == next_iter:
                    # Create list of tuples of (LAMMPS_ID, atom) so that atoms are
                    # reordered based on LAMMPS_ID
                    lines = []
                    for _ in range(n_atoms):
                        line = file_handler.readline().split()
                        # convert id to int
                        line[i_id] = int(line[i_id])
                        lines.append(line)
                    # sort list based on id
                    lines = sorted(lines, key=lambda x: x[i_id])

                    atoms = []
                    for line in lines:
                        if not atom_IDs or line[i_id] in atom_IDs:
                            atoms.append(create_atom(line))

                    configs.append(TemporalConfiguration(time_step, *atoms))

                    next_iter = config_indexes.next()
                config_iter += 1
                if config_iter >= stop:
                    break


            line = file_handler.readline()
    return Trajectory(*configs)

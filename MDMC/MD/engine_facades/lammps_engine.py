"""Facade for LAMMPS MD engine

This is a facade to PyLammps (added in 30th-Jul-2016 version), a convenience
wrapper for the LAMMPS Python interface i.e. where Python is extended with
LAMMPS.

Defining all interaction types requires that LAMMPS was built with the MOLECULE
package.

Notes
-----
When variables are either passed to or from PyLammps, the ctypes
conversion can mean that they are unnecessarily cast, particularly from float to
int.  This can cause issues as LAMMPS requires certain variables, e.g. number of
steps, to be int.  Therefore it is always a good idea to be cast these variables
when they are read from PyLammps e.g. int(lmp.variables['steps'].value).

A minor bug in LAMMPS (Dec 2018 version) means that nangletypes returned
by PyLammps is incorrectly set to ndihedraltypes."""

from copy import copy
from itertools import chain, combinations, count, product, tee
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


class PyLammpsAttribute:

    """
    A class which has a PyLammps object as an attribute

    It possesses attributes and methods relating to the PyLammps object

    Parameters
    ----------
    lmp : PyLammps, optional
        Set the lmp attribute to a PyLammps object. Default is None, which
        results in a new PyLammps object being initialised.
    atom_style : str, optional
        The LAMMPS atom_style, which determines the properties that can be
        associated which the atoms (e.g. charge, bonds). Default is 'full'.

    Attributes
    -----------
    lmp : PyLammps
        The PyLammps object owned by this class
    """

    def __init__(self, lmp=None, atom_style='full'):

        if lmp:
            self.lmp = lmp
        else:
            self.lmp = PyLammps()
            self.lmp.units('real')
            self.lmp.atom_style(atom_style)

    @property
    def system_state(self):

        """
        Get the PyLammps wrapper system state dictionary

        Returns
        -------
        System
                Contains the properties of the simulation box.
        """

        return self.lmp.system

    @property
    def fixes(self):

        """
        Get the PyLammps wrapper list of fixes

        Returns
        -------
        list of dict
            Each dict states the group, name and style of a LAMMPS fix which is
            applied
        """

        return self.lmp.fixes

    @property
    def fix_styles(self):

        """
        Get the styles of the fixes applied in LAMMPS

        Returns
        -------
        list of str
            The styles of the fixes
        """

        return [fix['style'] for fix in self.fixes]

    @property
    def fix_names(self):

        """
        Get the names of the fixes applied in LAMMPS

        Returns
        -------
        list of str
            The names of the fixes
        """

        return [fix['name'] for fix in self.fixes]


class LAMMPSEngine(PyLammpsAttribute, MDEngine):

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
    """

    @property
    def saved_config(self):

        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        array
            The configuration from the start of the run. Each row of the array
            corresponds to the LAMMPS atom ID - 1 (offset is due to array zero
            indexing) and the columns of the array are the x, y, z components of
            the position, the mass and the charge of each atom.
        """

        return self._saved_config

    @property
    def time_step(self):

        """
        Get or set the simulation time step in fs

        Returns
        -------
        float
            Simulation time step in fs
        """

        return self.lmp_simulation.time_step

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value):

        self.lmp_simulation.time_step = value

    @property
    def traj_step(self):

        """
        Get or set the number of simulation steps between saving the trajectory

        Returns
        -------
        int
            Number of simulation steps that elapse between the trajectory being
            stored
        """

        return self.lmp_simulation.traj_step

    @traj_step.setter
    def traj_step(self, value):

        self.lmp_simulation.traj_step = value

    @property
    def temperature(self):

        """
        Get or set the temperature of the simulation in K

        Returns
        -------
        float
            Temperature in K
        """

        return self.lmp_simulation.temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value):

        self.lmp_simulation.temperature = value

    @property
    def pressure(self):

        """
        Get or set the pressure of the simulation in atm

        Returns
        -------
        float
            Pressure in atm
        """

        return self.lmp_simulation.pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value):

        self.lmp_simulation.pressure = value

    @property
    def ensemble(self):

        """
        Get or set the ensemble object which applies a thermostat or barostat to
        LAMMPS

        Returns
        -------
        Ensemble
            The simulation ensemble
        """

        return self.lmp_simulation.ensemble

    @ensemble.setter
    def ensemble(self, value):

        self.lmp_simulation.ensemble = value

    @property
    def thermostat(self):

        """
        Get or set the string which specifies the thermostat

        Returns
        -------
        str
            The thermostat name
        """

        return self.ensemble.thermostat

    @thermostat.setter
    def thermostat(self, value):

        self.ensemble.thermostat = value

    @property
    def barostat(self):

        """
        Get or set the string which specifies the barostat

        Returns
        -------
        str
            The barostat name
        """

        return self.ensemble.barostat

    @barostat.setter
    def barostat(self, value):

        self.ensemble.barostat = value

    def setup_universe(self, universe, **settings):

        """
        Creates the simulation box, the atomic configuration, and the topology
        in LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC Universe which will be setup in LAMMPS.
        **settings
            atom_style : str
                A LAMMPS atom_style string. The default setting of 'real'
                will generally be appropriate.
        """

        super(LAMMPSEngine, self).__init__(atom_style=settings.get('atom_style',
                                                                   'full'))
        self.universe = universe
        self.lmp_universe = LAMMPSUniverse(self.universe, self.lmp, **settings)
        self._saved_config = None

    def setup_simulation(self, **settings):

        """
        Sets simulation parameters in LAMMPS, such as the thermodynamic
        variables, thermostat/barostat parameters and trajectory settings

        Parameters
        ----------
        **settings
            Passed to LAMMPSSimulation
        """

        self.lmp_simulation = LAMMPSSimulation(self.universe, self.lmp,
                                               **settings)

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
            self.ensemble.remove_ensemble_fixes()
            self.lmp_universe.apply_constraints()
            self.ensemble.apply_ensemble_fixes()


    def run(self, n_steps, equilibration=False):
        if not equilibration:
            # Remove previous dumps if they exist
            if 'traj1' in [dump['name'] for dump in self.lmp.dumps]:
                self.lmp.undump('traj1')
            # Store the trajectory in a NamedTemporaryFile
            self.trajectory_file = NamedTemporaryFile()
            # Custom trajectory output just saves the atom ID, type and
            # positions
            self.lmp.dump('traj1', 'all', 'custom', self.traj_step,
                          self.trajectory_file.name, 'id', 'type', 'x', 'y',
                          'z')
        else:
            reset_to_nve = False
            if self.thermostat is self.barostat is None:
                # If NVE ensemble, add a berendsen thermostat for equilibration
                reset_to_nve = True
                self.thermostat = 'berendsen'

        self.lmp.run(n_steps)

        if equilibration and reset_to_nve:
            self.thermostat = None

    def convert_trajectory(self):

        return convert_trajectory(self.trajectory_file,
                                  self.lmp_universe.atom_type_properties,
                                  time_step=self.time_step,
                                  universe=self.universe)

    def update_parameters(self):

        self.lmp_universe.update_parameters()

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

        self.lmp_universe.set_config(self.saved_config)


class LAMMPSUniverse(PyLammpsAttribute):
    # Class has to maintain a lot of state (attributes) as PyLammps class does
    # not
    #pylint: disable=too-many-instance-attributes

    """
    A class with what would be the equivalent in LAMMPS to the universe (i.e.
    the configuration and topology)

    Parameters
    ----------
    universe : Universe
        The MDMC Universe used to create the LAMMPSUniverse
    lmp : PyLammps, optional
        Set the lmp attribute to a PyLammps object. Default is None, which
        results in a new PyLammps object being initialised.
    **settings
        atom_style : str
            A LAMMPS atom_style string. The default setting of 'real' will
            generally be appropriate.
        nonbonded_mix : str
            The name of the formula which determines non-bonded mixing

    Attributes
    ----------
    universe : Universe
        The MDMC Universe which has been converted to this LAMMPSUniverse.
    atom_dict : dict
        A dictionary with {MDMC_atom: LAMMPS_atom}, where MDMC_atom is an MDMC
        Atom object and LAMMPS_atom is the corresponding LAMMPS Atom object.
    atom_types : dict
        A dictionary with {type_ID: MDMC_atom_group}, where the type_ID is a
        unique integer and MDMC_atom_group is a list of atoms which are
        identical in terms of element and interactions.
    atom_type_properties : list of tuples
        Each tuple is (symbol, mass) for all atom_types (ordered) by atom_type,
        where symbol is a string specifying the element of the atom_type and
        mass is a float specifying the mass of the atom_type.
    bonds : list of Bonds
        All Bond interactions in the MDMC universe.
    angles : list of BondAngles
        All BondAngle interactions in the MDMC universe.
    couls : list of Coulombics
        All Coulomobc interactions in the MDMC universe.
    disps : list of Dispersions
        All Dispersion interactions in the MDMC universe.
    bond_ID : dict
        A dictionary of {bond: ID pairs} relating each Bond object to a LAMMPS
        ID.
    angle_ID : dict
        A dictionary of {angle: ID pairs} relating each BondAngle object to
        a LAMMPS ID.
    """

    def __init__(self, universe, lmp=None, **settings):

        super(LAMMPSUniverse, self).__init__(lmp, settings.get('atom_style',
                                                               'full'))
        self.universe = universe
        self.atom_dict = {}
        self.atom_types = {}
        self.atom_type_properties = []

        self.bonds = []
        self.angles = []
        self.couls = []
        self.disps = []
        # ID is an acronym
        #pylint: disable=invalid-name
        self.bond_ID = {}
        self.angle_ID = {}
        self.nonbonded_mix = None

        self._define_simulation_box(self.universe)
        self._build_config(self.universe)
        self._add_topology(self.universe, **settings)
        self.update_parameters()

    @property
    def nonbonded_mix(self):

        """
        Get or set the formula used to calculate nonbonded interactions between
        different atom types

        Options are geometric, arithmetic and sixthpower, which are defined in
        the LAMMPS documentation.

        Returns
        -------
        str
            the name of the formula which determines non-bonded mixing

        Raises
        ------
        ValueError
            `str` specifies an unsupported mix name
        """

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

    def update_parameters(self):

        """
        Updates the LAMMPS force field parameters from the MDMC universe
        """

        self._update_charges()
        self._update_bonds(self.bonds)
        self._update_angles(self.angles)
        self._update_dispersions(self.universe)

    def _define_simulation_box(self, universe):

        """
        Defines a region and creates a simulation box that fills this region

        Parameters
        ----------
        universe : Universe
            The MDMC Universe used to create the region and simulation box.
        """

        # ID is an acronym
        #pylint: disable=invalid-name
        region_ID = 'universe'
        self._create_lammps_region(universe, region_ID)
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

    # ID is an acronym
    #pylint: disable=invalid-name
    def _create_lammps_region(self, universe, region_ID):

        """
        Create a geometry of the simulation box in LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC Universe used to create the region.
        region_ID : str
            The LAMMPS region ID
        """

        xlo = ylo = zlo = 0.
        xhi, yhi, zhi = universe.dims
        region_ID = 'universe'
        # 'block' gives a cuboidal universe
        self.lmp.region(region_ID, 'block', xlo, xhi, ylo, yhi, zlo, zhi,
                        units='box')

    def _build_config(self, universe):

        """
        Adds atoms to LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC Universe used to fill the LAMMPS box with atoms.
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

    def set_config(self, config):

        """
        Changes the positions of all of the atoms in the LAMMPS wrapper

        Parameters
        ----------
        config : array
            The positions, mass and charge of the atoms, used to set the LAMMPS
            configuration. Each row of the array must correspond to the LAMMPS
            atom ID - 1 (offset is due to array zero indexing) and the columns
            of the array must be the x, y, z components of the position, the
            mass and the charge of each atom.

        Raises
        ------
        IndexError
            If `config` does not contain the same number of atoms as LAMMPS
            possesses.
        """

        # Raise an IndexError if the config is not the correct size
        n_atoms = self.system_state.natoms
        if len(config) != n_atoms:
            raise IndexError('the new configuration does not specify the'
                             ' correct number of atoms')

        # The LAMMPS wrapper does not allow the configuration to be updated
        # simply by setting all atoms. Instead the position of the atoms must be
        # reset.
        index_components = list(enumerate(['x', 'y', 'z']))
        # LAMMPS IDs start at 1, so are offset from config indexes
        for id_offset in range(n_atoms):
            for index, component in index_components:
                self.lmp.set('atom', id_offset+1, component,
                             config[id_offset][index])

    def _max_n_interaction(self, atoms, name):

        """
        Parameters
        ----------
        atoms : list of Atoms
        name : str
            An Interaction type, for example 'Bond'.

        Returns
        -------
        int
            The maximum number of interactions with a given name that any atom
            in `atoms` possesses
        """

        return max([len(list(filter(lambda i: i.name == name,
                                    atom.interactions)))
                    for atom in atoms])

    def _add_topology(self, universe, **settings):

        """
        Add the bonded and nonbonded interactions to LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC Universe used to define the topology.
        **settings
            nonbonded_mix : str
                The name of the formula which determines non-bonded mixing

        Raises
        ------
        NotImplementedError
            If `universe` contains an interaction type that has not been
            implemented in the LAMMPS facade
        """

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

        pair_styles, pair_coeff_cmds = self._pair_style_coeff_commands(universe)
        if pair_styles:
            # hybrid/overlay allows multiple pair_styles for same atom_type pair
            self.lmp.pair_style('hybrid/overlay', *pair_styles)
            self._update_charges()
            self.nonbonded_mix = settings.get('nonbonded_mix')
            self._update_dispersions(universe, pair_coeff_cmds)
            # Apply LAMMPS modifications to nonbonded interactions
            self._modify_nonbonded_styles(disps+couls)

        if bonds:
            # Set used to remove duplicate bond styles, which are not required
            # to be (and in fact cannot) be passed to LAMMPS hybrid bond_style
            self.lmp.bond_style('hybrid',
                                *set(tuple([parse_bonded_styles(b)
                                            for b in bonds])))
            self._create_bonds(bonds)

        if angles:
            # Set used to remove duplicate bond styles, which are not required
            # to be (and in fact cannot) be passed to LAMMPS hybrid angle_style
            self.lmp.angle_style('hybrid',
                                 *set(tuple([parse_bonded_styles(a)
                                             for a in angles])))
            self._create_angles(angles)

        if self.universe.constraint_algorithm:
            self.apply_constraints()

    def _pair_style_coeff_commands(self, universe):

        """
        Parses all the NonBondedInteractions for every appropriate combination
        of atom_type pairs in an MDMC Universe, returning the correctly
        formatted input for pair_style and pair_coeff LAMMPS commands.

        Parameters
        ----------
        universe : Universe
            The MDMC Universe containing the NonBondedInteractions to
            be parsed.

        Returns
        -------
        pair_styles : list of str
            A flattened list of pair_style commands to be set in the
            LAMMPS interface.
        pair_coeff_cmds : list of str
            A list of pair_coeff commands for each atom_type pair to be
            set individually in the LAMMPS interface.
        """

        pair_styles = []
        pair_coeff_cmds = []
        for pair, inters in universe.nbis_by_atom_type_pairs.items():
            # Generates list of tuples containing styles and cutoffs.
            nb_styles = parse_all_nonbonded_styles(inters)
            # Generates list of tuples that contain each pair_coeff command.
            coeff_cmds = parse_dispersion_coefficients(inters, nb_styles)
            for idx, cmd in enumerate(coeff_cmds):
                coeff_cmds[idx] = (' '.join(str(type) for type in pair)
                                   + ' ' + cmd)
            pair_styles += nb_styles
            pair_coeff_cmds += coeff_cmds

        # Remove duplicates in pair_styles, create flattened list.
        pair_styles = list(chain.from_iterable(list(set(pair_styles))))

        return pair_styles, pair_coeff_cmds

    def _update_charges(self):

        """
        Updates the charges in LAMMPS
        """

        for atom, lmp_atom in self.atom_dict.items():
            try:
                self.lmp.set('atom',
                             lmp_atom.id,
                             'charge',
                             convert_unit(atom.charge))
            except AttributeError:
                raise AttributeError('LAMMPS requires all atoms in the universe'
                                     ' to have a charge.')

    def _update_dispersions(self, universe, pair_coeff_cmds=None):

        """
        Updates dispersion interactions in LAMMPS

        Parameters
        ----------
        universe : Universe
            The MDMC Universe containing NonBondedInteractions used to generate
            the pair_coeff commands if pair_coeff_cmds is not passed.
        pair_coeff_cmds : list of str
            A list of pair_coeff commands for each atom_type pair to be
            set individually in the LAMMPS interface.
            NOTE: the Coulombics for the appropriate atom_type pairs are also
            set in this method.
        """

        if not pair_coeff_cmds:
            _, pair_coeff_cmds = self._pair_style_coeff_commands(universe)

        for cmd in pair_coeff_cmds:
            self.lmp.pair_coeff(cmd)

    def _modify_nonbonded_styles(self, nonbonded_interactions):

        """
        Applies modifications to nonbonded pair styles, such as the VdW tail
        correction or setting a mixing style for interactions acting on unlike
        atom type pairs

        Parameters
        ----------
        nonbonded_interactions : list of NonbondedInteractions
            NonBondedInteractions which will have modifications applied to the
            corresponding pair styles.

        Warns
        -----
        warnings.warn
            If a pair style is specified which cannot have a vdw tail correction
            applied
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
                    # Only dispersion interactions can have vdw tail corrections
                    if inter.name == 'Dispersion' and inter.vdw_tail_correction:
                        if style in excluded:
                            warnings.warn('{0} pair style cannot have a'
                                          ' vdw tail correction'
                                          ' applied'.format(style))
                        else:
                            inter_str += ('tail', 'yes')
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

        Parameters
        ----------
        bonds : list of Bonds
            Bond interactions which will be created in LAMMPS.
        """

        special = 'no'
        # If bonds already exist, new bond IDs are generated from lowest unused
        # integer
        if self.bond_ID:
            start = max(self.bond_ID.values()) + 1
        else:
            start = 1
        for ID, bond in enumerate(bonds, start=start):
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

        Parameters
        ----------
        bonds : list of Bonds
            Bond interactions which will be updated in LAMMPS.
        """

        for bond in bonds:
            self.lmp.bond_coeff(self.bond_ID[bond],
                                *parse_bonded_coefficients(bond))

    def _create_angles(self, angles):

        """
        Creates coefficients and angles in LAMMPS, and fills the angle_ID
        dictionary with angle: ID pairs

        Parameters
        ----------
        angles : list of BondAngles
            BondAngle interactions which will be created in LAMMPS
        """

        special = 'no'
        # If bonds already exist, new bond IDs are generated from lowest unused
        # integer

        if self.angle_ID:
            start = max(self.angle_ID.values()) + 1
        else:
            start = 1
        for ID, angle in enumerate(angles, start=start):
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

        Parameters
        ----------
        angles : list of BondAngles
         BondAngle interactions which will be updated in LAMMPS
        """

        for angle in angles:
            self.lmp.angle_coeff(self.angle_ID[angle],
                                 *parse_bonded_coefficients(angle))

    def apply_constraints(self):

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
        # angles - the fix will be applied to this group. Applying constraint
        # fix only to this group improves performance.
        # chain is used to flatten inter.atoms, which is a list of tuples
        atom_types = set([atom.atom_type for inter in bonds+angles
                          for atom in chain.from_iterable(inter.atoms)])
        constrain_group = 'constrain_group'
        self.lmp.group(constrain_group, 'type', *atom_types)
        self.lmp.fix('constrain', constrain_group, *algorithm)


class LAMMPSSimulation(PyLammpsAttribute):
    # Class has to maintain a lot of state (attributes) as PyLammps class does
    # not
    #pylint: disable=too-many-instance-attributes

    """
    The attributes and methods related running a simulation in LAMMPS using a
    LAMMPSUniverse object

    Parameters
    ----------
    universe : Universe
        The MDMC Universe used to create the LAMMPSUniverse.
    lmp : PyLammps, optional
        Set the lmp attribute to a PyLammps object. Default is None, which
        results in a new PyLammps object being initialised.
    **settings
        temperature : float
        time_step : float
        traj_step : int
        skin : float
        neighbor_steps : int
        remove_linear_momentum : int
        remove_angular_momentum : int

    Attributes
    ----------
    universe : Universe
        An MDMC Universe object.
    traj_step : int
        Number of simulation steps that elapse between the trajectory being
        stored.
    ensemble : Ensemble
        Simulation ensemble, which applies a thermostat and barostat.
    """

    def __init__(self, universe, lmp=None, **settings):

        super(LAMMPSSimulation, self).__init__(lmp, settings.get('atom_style',
                                                                 'full'))
        self.universe = universe
        self.ensemble = Ensemble(self.lmp, **settings)
        self.temperature = settings.get('temperature')
        self.time_step = settings.get('time_step', 1.0)
        self.traj_step = settings['traj_step']

        self.skin = settings.get('skin', 2.0)
        self.neighbor_steps = settings.get('neighbor_steps', 1)

        # Setting _lin_momentum_steps and _ang_momentum_steps allows
        # _set_momentum_removers to be called when setting
        # self.lin_momentum_steps and self.ang_momentum_steps
        self._lin_momentum_steps = None
        self._ang_momentum_steps = None
        self.lin_momentum_steps = settings.get('remove_linear_momentum', 1)
        self.ang_momentum_steps = settings.get('remove_angular_momentum')

        self._set_kspace_solver()

    @property
    def time_step(self):

        """
        Get or set the simulation time step in fs

        Returns
        -------
        float
            Simulation time step in fs
        """

        return self._time_step

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value):

        self._time_step = value
        try:
            # Set the timestep in LAMMPS wrapper
            self.lmp.timestep(convert_unit(self._time_step))
        except AttributeError:
            pass

    @property
    def temperature(self):

        """
        Get or set the temperature of the simulation in K

        Returns
        -------
        float
            Temperature in K
        """

        return self._temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value):

        self._temperature = value
        self.ensemble.temperature = value
        try:
            # Set the initial temperature in the LAMMPS wrapper
            if self.system_state.natoms > 0:
                self.lmp.velocity('all', 'create',
                                  convert_unit(self._temperature),
                                  randint(1, 9999))
        except AttributeError:
            pass

    @property
    def pressure(self):

        """
        Get or set the pressure of the simulation in atm

        Returns
        -------
        float
            Pressure in atm
        """

        return self.ensemble.pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value):

        self.ensemble.pressure = value

    @property
    def thermostat(self):

        """
        Get or set the string which specifies the thermostat

        Returns
        -------
        str
            The thermostat name
        """

        return self.ensemble.thermostat

    @thermostat.setter
    def thermostat(self, value):

        self.ensemble.thermostat = value

    @property
    def barostat(self):

        """
        Get or set the string which specifies the barostat

        Returns
        -------
        str
            The barostat name
        """

        return self.ensemble.barostat

    @barostat.setter
    def barostat(self, value):

        self.ensemble.barostat = value

    @property
    def skin(self):

        """
        Get or set the skin distance in Ang

        Returns
        -------
        float
            The skin distance in Ang. This distance plus the force cutoff
            distance determines which atom pairs are stored in the neighbor
            list.
        """

        return self._skin

    @skin.setter
    @unit_decorator(unit=units.LENGTH)
    def skin(self, value):

        self._skin = value
        # Set the neighor list parameters in the LAMMPS wrapper
        self.lmp.neighbor(convert_unit(self._skin), 'bin')


    @property
    def neighbor_steps(self):

        """
        Get or set the number of steps between neighbor list updates

        Returns
        -------
        int
            Number of steps between neighbor list updates
        """

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
    def lin_momentum_steps(self):

        """
        Get or set the number of steps between resetting the linear momentum

        Returns
        -------
        int
            Number of steps between the linear momentum being removed
        """

        return self._lin_momentum_steps

    @lin_momentum_steps.setter
    def lin_momentum_steps(self, value):

        self._lin_momentum_steps = value
        # Set the momentum removers in the LAMMPS wrapper
        self._set_momentum_removers()

    @property
    def ang_momentum_steps(self):

        """
        Get or set the number of steps between resetting the angular momentum

        Returns
        -------
        int
            Number of steps between the angular momentum being removed
        """

        return self._ang_momentum_steps

    @ang_momentum_steps.setter
    def ang_momentum_steps(self, value):

        self._ang_momentum_steps = value
        # Set the momentum removers in the LAMMPS wrapper
        self._set_momentum_removers()

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

        if self.system_state.natoms > 0:
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

    def _set_kspace_solver(self):

        """
        Creates a k-space solver in LAMMPS using kspace_style, if one is
        required

        Uses either the kspace_solver, the electrostatic_solver or both the
        electrostatic_solver and dispersive_solver attribute of the MDMC
        universe to set the kspace_style. Note that LAMMPS does not support
        different electrostatic and dispersive solvers. Setting with equivalent
        electrostatic and dispserive solvers is equivalent to setting with
        kspace_solver. LAMMPS also does not support just applying a dispersive
        solver.

        Raises
        ------
        TypeError
            If `self.universe` has both a `kspace_solver` and either an
            `electrostatic_solver` or a `dispersive_solver`.
        TypeError
            If `self.universe` has a different `electrostatic_solver` and
            `dispersive_solver`.
        TypeError
            If `self.universe` only has a `dispersive_solver`.
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


class Ensemble(PyLammpsAttribute):
    # Class has to maintain a lot of state (attributes) as PyLammps class does
    # not
    #pylint: disable=too-many-instance-attributes

    """
    A thermodynamic ensemble determined by applying a thermostat and/or barostat

    Parameters
    ----------
    lmp : PyLammps
        Set the lmp attribute to a PyLammps object.
    temperature : float, optional
        Thermostat temperature. Default is None, which is only valid if a
        thermostat is also None.
    pressure : float, optional
        Barostat pressure. Default is None, which is only valid if a barostat is
        also None.
    thermostat : str
        Name of a thermostat to be applied.
    barostat : str
        Name of a barostat to be applied.
    **settings
        time_step : float
        t_damp : int
        p_damp : int
        t_window : float
        t_fraction : float
        rescale_step : int

    Attributes
    ----------
    rescale_step : int
        Number of steps between applying temperature rescaling. This only
        applies to rescale thermostats.
    """

    def __init__(self, lmp, temperature=None, pressure=None, thermostat=None,
                 barostat=None, **settings):

        # Requires a lmp object as thermostats cannot be applied before
        # configuration is defined
        super(Ensemble, self).__init__(lmp)
        # Setting _thermostat and _barostat allows apply_ensemble_fixes to be
        # called when setting self.temperature and self.pressure
        self._thermostat = None
        self._barostat = None
        self.temperature = temperature
        self.pressure = pressure

        self.time_step = settings.get('time_step', 1.0)
        self.t_damp = settings.get('t_damp', 100)
        self.p_damp = settings.get('p_damp', 1000)
        self.t_window = settings.get('t_window')
        self.t_fraction = settings.get('t_fraction')
        self.rescale_step = settings.get('rescale_step')

        self.thermostat = thermostat
        self.barostat = barostat

    @property
    def time_step(self):

        """
        Get or set the simulation time step in fs
        """

        return self._time_step

    @time_step.setter
    @unit_decorator(unit=units.TIME)
    def time_step(self, value):

        self._time_step = value

    @property
    def temperature(self):

        """
        Get or set the temperature of the simulation in K
        """

        return self._temperature

    @temperature.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def temperature(self, value):

        self._temperature = value
        self.apply_ensemble_fixes()

    @property
    def pressure(self):

        """
        Get or set the pressure of the simulation in atm
        """

        return self._pressure

    @pressure.setter
    @unit_decorator(unit=units.PRESSURE)
    def pressure(self, value):

        self._pressure = value
        self.apply_ensemble_fixes()

    # Unit has to be applied to getter due to operation in setter
    @property
    @unit_decorator_getter(unit=units.TIME)
    def t_damp(self):

        """
        Get or set the number of time steps over which the temperature is
        relaxed

        Required for Nose-Hoover, Berendsen and Langevin thermostats.

        Returns
        -------
        int
            Number of time steps

        Raises
        ------
        AttributeError
            If `self.time_step` has not been set.
        """

        # t_damp is stored in units of time - convert back to number of steps
        # here
        try:
            return self._t_damp / self.time_step
        except TypeError:
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

        """
        Get or set the number of time steps over which the pressure is
        relaxed

        Required for Nose-Hoover or Berendsen barostats. The `time_step` must
        have been set before `p_damp`.

        Returns
        -------
        int
            Number of time steps

        Raises
        ------
        AttributeError
            If `self.time_step` has not been set.
        """

        # p_damp is stored in units of time - convert back to number of steps
        # here
        try:
            return self._p_damp / self.time_step
        except TypeError:
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

        """
        Get or set the fraction by which the temperature is rescaled to the
        target temperature

        This is required for the rescale thermostat.

        Returns
        -------
        float
            Fraction (i.e. between 0.0 and 1.0 inclusive) by which the
            temperature is rescaled

        Raises
        ------
        ValueError
            If set to a value outside of 0.0 and 1.0 inclusive.
        """

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

        """
        Get or set the temperature range in K in which the temperature is not
        rescaled

        This only applies to rescale thermostats.

        Returns
        -------
        float
            temperature range in K
        """

        return self._t_window

    @t_window.setter
    @unit_decorator(unit=units.TEMPERATURE)
    def t_window(self, value):

        self._t_window = value

    @property
    def thermostat(self):

        """
        Get or set the string which specifies the thermostat

        Raises
        ------
        AttributeError
            If `self.temperature` has not been set.
        """

        return self._thermostat

    @thermostat.setter
    def thermostat(self, value):

        if value and not self.temperature:
            raise AttributeError('all ensembles with a thermostat must have a'
                                 ' temperature')
        self._thermostat = value
        # Set the thermostat and barostat in LAMMPS wrapper
        self.apply_ensemble_fixes()

    @property
    def barostat(self):

        """
        Get or set the string which specifies the barostat

        Raises
        ------
        AttributeError
            If `self.pressure` has not been set.
        """

        return self._barostat

    @barostat.setter
    def barostat(self, value):

        if value and not self.pressure:
            raise AttributeError('all ensembles with a barostat must have a'
                                 ' pressure')

        self._barostat = value
        # Set the thermostat and barostat in LAMMPS wrapper
        self.apply_ensemble_fixes()

    def remove_ensemble_fixes(self):

        """
        Removes all LAMMPS fixes relating to the ensemble i.e. removes all
        thermostats and barostats

        This must be done before thermostat and barostat fixes are added, so
        that there is no conflict with existing thermostat and barostats fixes.
        It is also required for Shake and Rattle fixes which cannot be added
        after barostat fixes have been applied.
        """

        for name in self.fix_names:
            if name in ['nve', 'nvt', 'npt', 'nph', 't_berendsen',
                        'p_berendsen', 'langevin', 'rescale']:
                self.lmp.unfix(name)

    def apply_ensemble_fixes(self):

        """
        Passes the required LAMMPS fixes to apply a specific thermodynamic
        ensemble to the simulation

        Removes all pre-existing thermostat and barostat fixes
        """

        self.remove_ensemble_fixes()

        if not self.thermostat and not self.barostat:
            self.lmp.fix('nve', 'all', 'nve')
        else:
            if self.thermostat:
                temp = convert_unit(self.temperature)
                if self.thermostat != 'rescale':
                    t_damp = convert_unit(self.t_damp)
                    thermo_params = [temp, temp, t_damp]
            if self.barostat:
                press = convert_unit(self.pressure)
                p_damp = convert_unit(self.p_damp)
                press_params = ['iso', press, press, p_damp]

            # Apply thermostat
            if self.thermostat == 'nose':
                if self.barostat == 'nose':
                    self.lmp.fix('npt', 'all', 'npt', 'temp',
                                 *thermo_params + press_params)
                else:
                    self.lmp.fix('nvt', 'all', 'nvt', 'temp', *thermo_params)
            elif self.thermostat == 'berendsen':
                # berendsen does not do time integration so also requires nve
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('t_berendsen', 'all', 'temp/berendsen',
                             *thermo_params)
            elif self.thermostat == 'langevin':
                # langevin does not do time integration so also requires nve
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('langevin', 'all', 'langevin',
                             *thermo_params + [randint(0, 9999)])
            elif self.thermostat == 'rescale':
                # temp/rescale does not do time integration so also requires nve
                t_window = convert_unit(self.t_window)
                self.lmp.fix('nve', 'all', 'nve')
                self.lmp.fix('rescale', 'all', 'temp/rescale',
                             self.rescale_step, temp, temp, t_window,
                             self.t_fraction)
            # Apply barostat
            if self.barostat == 'berendsen':
                self.lmp.fix('p_berendsen', 'all', 'press/berendsen',
                             *press_params)
            elif self.barostat == 'nose' and self.thermostat != 'nose':
                if 'nve' in self.fix_names:
                    self.lmp.unfix('nve')
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
    'ENERGY':units.Unit('kcal') / units.Unit('mol'),
    'FORCE':units.Unit('kcal') / (units.Unit('Ang') * units.Unit('mol')),
    'PRESSURE':units.Unit('atm')
}


def convert_unit(value, unit=None, to_lammps=True):

    """
    Converts between MDMC units and LAMMPS real units

    Parameters
    ----------
    value : array_like or float_like
        The value of the physical property to be converted, in MDMC units.
        Must derive from either ndarray or float.
    unit : Unit, optional
        The unit of the value. If None, the value must possess a unit attribute
        i.e. derive from UnitFloat or UnitArray. Default is None.
    to_lammps : bool, optional
        If True the conversion is from MDMC units to LAMMPS units. Default is
        True.

    Returns
    -------
    float or array
        Value in LAMMPS units if to_lammps is True, otherwise value in MDMC
        units. Return type is same as `value` type.
    """

    def expand_components(unit, system):

        """
        Expands out the components of a unit, so that the unit is expressed
        purely in terms of base units. The only exception to this is units
        which occur in system: these are kept in the list of components.

        Parameters
        ----------
        unit : Unit
            The unit to be expanded out
        system : dict
            A dict of {Property: Unit} pairs, which is used to substitute system
            units into the expanded components.

        Returns
        -------
        tuple
            A tuple of (num, denom), where num is a list of all base units in
            the numerator, and denom is a list of all base units in the
            denominator
        """

        def is_sublist_of_list(sub, lst):

            """
            Determines if all of the elements in a sublist are in a list,
            including ensuring that any duplicates in the sublist have at least
            the same number of duplicates in the list

            Parameters
            ----------
            sub : list
                The sublist to be tested
            lst : list
                The list the sub is tested against

            Returns
            -------
            bool
                True if all of the elements of sub are in ls, and False if not
            """

            return all(sub.count(x) <= lst.count(x) for x in set(sub))

        def remove_components(remove_comps, comps):

            """
            Removes all elements of a list of components from another list of
            components

            Parameters
            ----------
            remove_comps : list
                The components to be removed
            comps : list
                The components from which remove_comps is removed

            Returns
            -------
            list
                A list of components from which remove_comps have been removed
            """

            for remove_comp in remove_comps:
                comps.remove(remove_comp)

            return comps

        num, denom = [], []
        # If unit is in system of units, don't break down to components, as
        # a unit in the system of units always has a conversion, even if it is
        # a compound unit.
        if unit.base or unit in system.values():
            num.append(unit)
        else:
            # tuple unpacking style below appends to num and denom lists
            for comp in unit.components['numerator']:
                num[len(num):], denom[len(denom):] = expand_components(comp,
                                                                       system)
            for comp in unit.components['denominator']:
                denom[len(denom):], num[len(num):] = expand_components(comp,
                                                                       system)

            # Substitute in units rather than separated out components if a unit
            # exists in the system of units.  This is because the conversion can
            # always be determined for units in the system of units. Base units
            # are filtered as they can only replace themselves (as they are
            # their only component).
            for sys_unit in (unit for unit in system.values() if not unit.base):
                # while is required for powers of sys_unit, as otherwise only
                # a single power will be removed
                while True:
                    sys_unit_num = sys_unit.components['numerator']
                    sys_unit_denom = sys_unit.components['denominator']
                    # Determine if all of the components of unit are in the
                    # numerator and denominator lists. If so, remove them and
                    # replace with the unit in the numerator.
                    if (is_sublist_of_list(sys_unit_num, num) and
                            is_sublist_of_list(sys_unit_denom, denom)):
                        num = remove_components(sys_unit_num, num)
                        denom = remove_components(sys_unit_denom, denom)
                        num.append(sys_unit)
                    # Do the same for the inverse (i.e. unit's numberator
                    # components in the denominator list and vice versa). If so,
                    # remove them and replace with the unit in the denominator.
                    elif (is_sublist_of_list(sys_unit_num, denom) and
                          is_sublist_of_list(sys_unit_denom, num)):
                        denom = remove_components(sys_unit_num, denom)
                        num = remove_components(sys_unit_denom, num)
                        denom.append(sys_unit)
                    # Breaks if the components of sys_unit are not found in num
                    # and denom
                    else:
                        break
        return num, denom

    # If no unit argument is passed, the value must possess a unit
    if not unit:
        unit = value.unit
    # Expand the unit in terms of its base units (for numerator and denominator)
    if to_lammps:
        l_sys = copy(SYSTEM)
        # For angular potential strength LAMMPS requires the units in rad,
        # rather than degrees (which is uses otherwise). Therefore if the unit
        # is in MDMC angular potential strength units (energy / angle^2), the
        # ANGLE entry in SYSTEM is replaced by radians.
        if unit == units.SYSTEM['ENERGY'] / units.SYSTEM['ANGLE'] ** 2:
            l_sys['ANGLE'] = units.Unit('rad')

        expanded_unit = expand_components(unit, units.SYSTEM)
        system_inv = {unit:property for property, unit in units.SYSTEM.items()}
        # Apply inversion to all components
        unit_nums, unit_denoms = map(lambda comp_list: [l_sys[system_inv[comp]]
                                                        for comp in comp_list],
                                     expanded_unit)

        conv_nums, conv_denoms = [], []
        for component in unit_nums:
            conv_nums[len(conv_nums):], conv_denoms[len(conv_denoms):] = \
                expand_components(component, l_sys)
        for component in unit_denoms:
            conv_denoms[len(conv_denoms):], conv_nums[len(conv_nums):] = \
                expand_components(component, l_sys)
    else:
        conv_denoms, conv_nums = expand_components(unit, SYSTEM)

    for component in conv_nums:
        value /= getattr(units, component)
    for component in conv_denoms:
        value *= getattr(units, component)

    return value


def parse_bonded_styles(interaction):

    """
    Converts MDMC InteractionFunction names for BondedInteractions to LAMMPS
    bond styles

    Parameters
    ----------
    interaction : BondedInteraction
        BondedInteraction to be parsed into LAMMPS bond style.

    Returns
    -------
    str
        LAMMPS bond style corresponding to `interaction`

    Raises
    ------
    NotImplementedError
        If `interaction` has a function that has not been implemented in the
        LAMMPS facade.
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

    Parameters
    ----------
    interaction : NonBondedInteraction
        NonBondedInteraction to be parsed into LAMMPS pair style.

    Returns
    -------
    list of str
        LAMMPS pair style corresponding to `interaction`

    Raises
    ------
    NotImplementedError
        If `interaction` has a function that has not been implemented in the
        LAMMPS facade.
    """

    lmp_str = []
    if interaction.function_name == 'Buckingham':
        lmp_str.append('buck')
    elif interaction.function_name == 'LennardJones':
        lmp_str.append('lj')
    elif interaction.function_name == 'Coulomb':
        lmp_str.append('coul')
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    if interaction.cutoff:
        cutoff = convert_unit(interaction.cutoff)
        kspace = interaction.universe.kspace_solver
        electrostatic = interaction.universe.electrostatic_solver
        dispersive = interaction.universe.dispersive_solver
        if (kspace or (electrostatic and interaction.name == 'Coulombic')
            or (dispersive and interaction.name == 'Dispersion')):
            lmp_str[-1] += '/long'
        else:
            if interaction.function_name != 'Buckingham':
                lmp_str[-1] += '/cut'
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
    'buck', 'lj/cut', 'coul/cut' and 'coul/long' can all be passed separately,
    the dispersive and coulombic styles are combined (dispersive always
    preceding coulombic) as this is dealt with more efficiently in the LAMMPS
    engine. 'buck/long' and 'lj/long' only exist as part of other pair styles,
    such as 'buck/long/coul/long' and 'lj/long/coul/long', so must be combined.

    Parameters
    ----------
    interactions : list of NonBondedInteractions
        NonBondedInteractions to be parsed into LAMMPS pair styles.

    Returns
    -------
    list of tuple
        A list of all of the combined LAMMPS pair styles corresponding to
        `interactions`. Each tuple contains the combined pair_style as one
        element, and the cutoffs as the second element, for example:

            ('lj/cut/coul/cut', '5.0 10.0')

        If additional arguments are required when combining styles, they will
        be added and returned in the tuple as the third and middle element,
        for example:

            ('buck/long/coul/long', 'long long', '10.0')

    Raises
    ------
    ValueError
        If Dispersion and Coulombic interactions have different long range
        cutoffs, which is not implemented in LAMMPS.
    ValueError
        If long range dispersive interaction not defined in conjunction with
        a long range Coulombic interaction.
    """

    def remove_duplicates(interactions):

        """
        Removes duplicates from list of tuples whilst maintaining order passed.

        Parameters
        ----------
        interactions : list of tuple
            List of parsed nonbonded interactions, where each element is a
            tuple of the form (style, parameters), i.e. ('lj/cut', cutoff).

        Returns
        -------
        list of tuple
            List of tuples of the above form, with duplicates removed but
            order as passed in interactions maintained.
        """

        return [x for x in interactions if not (x in set() or set().add(x))]


    def check_validity(pair_style, cutoffs=None):

        """
        Tests the validity of a LAMMPS pair_style.

        Parameters
        ----------
        pair_style : str
            The LAMMPS pair_style to be checked.
        cutoffs : list of float
            The cutoff distances for each style in the pair_style.

        Returns
        -------
        list of str
            Contains the inputted pair_style (if valid) as well as any extra
            terms required in specific cases (i.e. 'long long' if the pair
            style buck/long/coul/long is passed).

        Raises
        ------
        ValueError
            If the pair_style passed is not valid.
        """

        if pair_style in ['buck/long/coul/cut', 'lj/long/coul/cut']:
            raise ValueError('Invalid pair_style: ' + pair_style
                             + '. LAMMPS requires long range Coulombics to be'
                             ' defined in conjunction with long range Dispersion'
                             ' interactions.')
        if pair_style in ['buck/long/coul/long', 'lj/long/coul/long']:
            if cutoffs[0] != cutoffs[1]:
                raise ValueError('LAMMPS requires both cutoffs to be the same'
                                 ' for long range Dispersion and Coulombic pair'
                                 ' styles.')
            return [pair_style, 'long long']

        return [pair_style]

    # parsed_interactions = list(set([tuple(parse_nonbonded_styles(nb))
    #                                 for nb in interactions]))
    parsed_interactions = remove_duplicates([tuple(parse_nonbonded_styles(nb))
                                             for nb in interactions])
    # Flatten tuples of form (style, parameters), i.e. ('lj/cut', cutoff).
    flat_interactions = list(chain.from_iterable(parsed_interactions))
    # Define all Dispersion and Coulombic styles currently supported
    # Dispersion styles always precede coulombic styles in LAMMPS pair styles
    disp_styles = ['buck', 'buck/long', 'lj/cut', 'lj/long']
    coul_styles = ['coul/cut', 'coul/long']

    lmp_str = []
    combined = []
    for d_style, c_style in product(disp_styles, coul_styles):
        if d_style in flat_interactions and c_style in flat_interactions:
            for int1, int2 in combinations(parsed_interactions, 2):
                if (int1[0] == d_style and int2[0] == c_style
                        or int1[0] == c_style and int2[0] == d_style):
                    combined.append(int1)
                    combined.append(int2)
                    indiv_cmd = []
                    for element in check_validity('/'.join([d_style, c_style]),
                                                  cutoffs=[int1[1], int2[1]]):
                        indiv_cmd.append(element)
                    if int1[0] == d_style:
                        cutoffs = str(int1[1])
                        if int2[1] != int1[1]:
                            cutoffs += ' ' + str(int2[1])
                    else:
                        cutoffs = str(int2[1])
                        if int2[1] != int1[1]:
                            cutoffs += ' ' + str(int1[1])
                    lmp_str.append(tuple(indiv_cmd + [cutoffs]))
    # Remove interactions from parsed interactions if already combined
    for parsed, used in product(parsed_interactions, combined):
        if parsed == used and parsed in parsed_interactions:
            parsed_interactions.remove(parsed)
    return lmp_str + parsed_interactions


def parse_bonded_coefficients(interaction):

    """
    Orders MDMC Parameters for input to LAMMPS bond_coeff and angle_coeff

    Parameters
    ----------
    interaction : BondedInteraction
        BondedInteraction where its style and parameters will be parsed.

    Returns
    -------
    list of str
        Style and parameters converted to the input format for LAMMPS bond_coeff
        and angle_coeff

    Raises
    ------
    NotImplementedError
        If `interaction` has a function that has not been implemented in the
        LAMMPS facade.
    """

    parameters = {p.name:convert_unit(p.value)
                  for p in interaction.params}
    style = parse_bonded_styles(interaction)

    if style == 'harmonic':
        ordered_parameters = [parameters['potential_strength'],
                              parameters['equilibrium_state']]
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    return [style] + ordered_parameters


def parse_dispersion_coefficients(interactions, nonbonded_styles=None):

    """
    Orders MDMC Parameters for input to LAMMPS pair_coeff

    Parameters
    ----------
    interactions : list of NonBondedInteraction
        NonBondedInteractions where their style and parameters will be parsed.
    nonbonded_styles : list of tuple
        The parsed NonBondedInteractions, where the combined pair styles have
        been created by the parse_all_nonbonded_styles function.
        If None (default), the interactions are parsed manually.

    Returns
    -------
    coeff_cmds : list of str
        Each element is a partial pair_coeff command, containing a pair_style,
        the ordered Dispersion parameters converted to the correct input format
        for LAMMPS pair_coeff, and the cutoffs.
        For example, if interactions contains a Buckingham with parameters
        A, B, C = 4.184, 1.0, 4.184, and a cutoff of 20.0, and a Coulombic with
        a cutoff of 10.0, the str element of the list will be:

            'buck/coul/cut 1.0 1.0 1.0 20.0 10.0'

    Raises
    ------
    NotImplementedError
        If `interaction` has a function that has not been implemented in the
        LAMMPS facade.
    ValueError
        If the LAMMPS Buckingham parameter rho is less than or equal to zero.
    """

    if not nonbonded_styles:
        nonbonded_styles = parse_all_nonbonded_styles(interactions)

    coul_styles = ['coul/cut', 'coul/long']
    coeff_cmds = []
    for style in nonbonded_styles:
        pair_style = style[0]
        cutoffs = style[-1]
        if 'buck' in pair_style:
            for inter in interactions:
                if inter.function.name == 'Buckingham':
                    params = {p.name:convert_unit(p.value)
                              for p in inter.params}
            ordered_params = [params['A'],
                              params['B'] ** -1,
                              params['C']]
            try:
                assert ordered_params[1] > 0
            except AssertionError:
                raise ValueError('LAMMPS Buckingham parameter rho (= 1 / B)'
                                 ' must be greater than 0')
            coeff_cmd = (pair_style + ' '
                         + ' '.join(str(p) for p in ordered_params) + ' '
                         + cutoffs)
        elif 'lj' in pair_style:
            for inter in interactions:
                if inter.function.name == 'LennardJones':
                    params = {p.name:convert_unit(p.value)
                              for p in inter.params}
            ordered_params = [params['epsilon'],
                              params['sigma']]
            coeff_cmd = (pair_style + ' '
                         + ' '.join(str(p) for p in ordered_params) + ' '
                         + cutoffs)
        elif pair_style in coul_styles:
            coeff_cmd = pair_style
        else:
            raise NotImplementedError('This InteractionFunction has not been'
                                      ' implemented in the LAMMPS facade')
        coeff_cmds.append(coeff_cmd)

    return coeff_cmds


def parse_kspace_solver(solver):

    """
    Converts an MDMC kspace solver for input to LAMMPS kspace_style

    Parameters
    ----------
    solver : KSpaceSolver
        A KSpaceSolver to be parsed.

    Returns
    -------
    list
        Style and parameters for input to LAMMPS kspace_style

    Raises
    ------
    NotImplementedError
        If `solver` type has has not been implemented in the LAMMPS facade.
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

def parse_constraint(constraint_algorithm, bonds=None, bond_ID_dict=None,
                     angles=None, angle_ID_dict=None):
    # ID is an acronym
    #pylint: disable=invalid-name

    """
    Converts an MDMC constraint algorithm for input to LAMMPS fix

    At least one of bonds and angles must be passed.

    Parameters
    ----------
    constraint_algorithm : ConstraintAlgorithm
        An object that derives from ConstraintAlgorithm to be parsed.
    bonds : list of Bonds, optional
        Constrained Bond interactions.
    bond_ID_dict : dict, optional
        Dictionary with {bond: ID pairs} relating each Bond object to a LAMMPS
        ID.
    angles : list of BondAngles, optional
        Constrained BondAngle interactions.
    angle_ID_dict : dict, optional
        Dictionary with {angle: ID pairs} relating each BondAngle object to a
        LAMMPS ID.

    Returns
    -------
    list
        Input parameters for LAMMPS fix, not including the first two terms
        (fix ID, group-ID).  The output list has a maximum length of 7, where
        the last four entries are optional but a minimum of two is required::

            [algorithm name, accuracy, max iterations, 'b', bond IDs,
             'a', angle IDs]

    Raises
    ------
    TypeError
        If there is not at least one constrained interaction passed.
    NotImplementedError
        If `constraint_algorithm` not been implemented in the LAMMPS facade.
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
        if not bond_ID_dict:
            bond_ID_dict = {}
        lmp_str.append('b')
        lmp_str += [bond_ID_dict[bond] for bond in bonds if bond.constrained]
    if angles:
        if not angle_ID_dict:
            angle_ID_dict = {}
        lmp_str.append('a')
        lmp_str += [angle_ID_dict[angle] for angle in angles
                    if angle.constrained]

    return lmp_str


def partition(items, predicate):

    """
    Partitions an iterable using a predicate

    Parameters
    ----------
    items : iterable
        An interable to be partitioned.
    predicate : function
        A predicate that can be applied to items to returned True or False.

    Returns
    -------
    tuple
        A tuple of (gen_true, gen_false), where gen_true is a generator of all
        items for which the predicate returned True, and gen_false is a
        generator of all items for which the predicate returned False
    """

    iter_a, iter_b = tee((predicate(item), item) for item in items)
    return ((item for pred, item in iter_a if pred),
            (item for pred, item in iter_b if not pred))


def partition_interactions(interactions, names, unpartitioned=False, lst=False):

    """
    Partitions an iterable of Interaction objects using a list of Interaction
    names

    This occurs by using partition to filter out one Interaction type for each
    loop, so previously identified Interactions are no longer considered.

    Parameters
    ----------
    interactions : iterable of Interactions
        An interable of Interaction objects to be partitioned.
    names : list of str
        Names of Interaction classes.
    unpartitioned : bool, optional
        If True, then a generator containing any Interaction objects that did
        not have a name in names is returned as an additional item in the tuple.
        Default is False.
    lst : bool, optional
        If True, then the returned tuple contains lists rather than generators.
        Interaction objects which have the name specified by names[n]. Default
        is False.

    Returns
    -------
    tuple
        A tuple of length len(names) where index n is a generator of all of the
        Interaction objects which have the name specified by `names[n]`. If
        `unpartitioned` is True, tuple is length n+1. If `lst` is True, the
        generators are replaced by lists.

    Example
    -------
    Partion interactions into Bonds and BondAngles::
        bonds, angles = partition_interactions(interactions,
                                               ['Bond, BondAngle'])
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


def convert_trajectory(trajectory_file, atom_type_properties, start=0,
                       stop=None, step=1, **settings):

    """
    Converts between a LAMMPS trajectory dump and an MDMC trajectory

    The LAMMPS dump must include at least id, atom_type, and xyz positions. The
    xyz positions must be consecutive and in that order. The same is true of the
    xyz components of the velocity, if they are provided.

    LAMMPS dump files do not include the time that elapses in a single
    simulation step (time_step) and so this must be passed in fs, otherwise it
    defaults to 1.0 fs.

    Parameters
    ----------
    trajectory_file : file
        A LAMMPS dump (trajectory) file.
    atom_type_properties : list of tuples
        Each tuple is (symbol, mass) for all atom_types (ordered) by atom_type,
        where symbol is a string specifying the element of the atom_type and
        mass is a float specifying the mass of the atom_type.
    start : int
        The index of the first trajectory, inclusive.
    stop : int
        The index of the last trajectory, exclusive.
    step : int
        The step size between trajectories.
    **settings
        time_step : float
            The simulation time step in fs
        scaled_positions : bool
            If the `trajectory_file` has scaled positions
        universe : Universe
            MDMC Universe against which to compare number of atoms in
            `trajectory_file`.
        atom_IDs : list
            LAMMPS IDs of the atoms which should be included. If not passed
            then all atoms are included in the converted trajectory.

    Returns
    -------
    Trajectory
        The MDMC Trajectory corresponding to the LAMMPS `trajectory_file`

    Raises
    ------
    AssertionError
        If `universe` is passed, and the number of atoms in the
        `trajectory_file` is not the same as in the `universe`.
    TypeError
        If `trajectory_file` describes a triclinic universe.

    Warns
    -----
    warnings.warn
        If no `time_step` is passed, it is set to 1.0 fs.
    """

    def create_atom(line):

        """
        Create an MDMC atom from a line in a LAMMPS dump (trajectory) file

        At a minimum it will set the ID, atom_type and position of the atom. It
        will also set the velocity if included in the line, and the universe, if
        this was passed to convert_trajectory().

        Parameters
        ----------
        line : array
            Array containing a line from the ATOMS sections of a LAMMPS dump
            file. The array must contain the atom ID, the atom type, and the
            x, y, and z (or scaled equivalents) components of the position,
            which are assumed to be adjacent. It will also set the velocity of
            the atom if this is included in the line.

        Returns
        -------
        Atom
            MDMC Atom object corresponding to the `line`
        """

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

    # Warn if no time_step is specified
    try:
        time_step = settings['time_step']
    except KeyError:
        warnings.warn("Trajectory has no time step, defaulting to 1.0 fs")
        time_step = 1.


    # Change expected position string if scaled positions are used
    pos_string = 'xs' if settings.get('scaled_positions', False) else 'x'

    universe = settings.get('universe')
    # ID is an acronym
    #pylint: disable=invalid-name
    atom_IDs = settings.get('atom_IDs')

    configs = []
    frame_n = start
    # Use count to create range so that stop can be undefined
    frame_indexes = count(start, step)
    # next_frame_n next attribute is assigned dynamically
    next_frame_n = next(frame_indexes) #pylint: disable=no-member
    with open(trajectory_file.name, 'r') as file_handler:
        line = file_handler.readline()
        while line:

            # LAMMPS TIMESTEP is the number of time steps that have elapsed. To
            # avoid confusion with time_step (the amount of time that elapses in
            # a single simulation step, i.e. dt), these are referred to as
            # frames.
            if 'ITEM: TIMESTEP' in line:
                line = file_handler.readline()
                frame = int(line.split()[0])

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
                # if universe:
                #     for i in range(3):
                #         line = file_handler.readline()
                #         dmin, dmax = [float(splt) for splt in line.split()]
                #         print(line)
                #         assert dmin == 0.0
                #         # unit is taken from universe dims (which is a
                #         # UnitArray)
                #         assert dmax == convert_unit(universe.dims[i],
                #                                     universe.dims.unit)

            if 'ITEM: ATOMS' in line:
                if frame_n == start:
                    # LAMMPS dump files contain order of LAMMPS atom properties,
                    # at each time step. As these should not change with time
                    # step only determine this order for first required time
                    # step. Assumes that position components (x y z) and
                    # velocity components (vx vy vz) are always adjacent and
                    # ordered as shown.
                    splt = line.split()
                    # Requires id, type and position to be defined, velocity is
                    # optional
                    i_id, i_type, i_pos = [splt.index(prop) - 2 for prop
                                           in ['id', 'type', pos_string]]
                    if 'vx' in splt:
                        i_vel = splt.index('vx')
                    else:
                        i_vel = None

                if frame_n == next_frame_n:
                    # Reads all atom lines before creating any atoms. By
                    # creating a list of tuples of (LAMMPS_ID, atom), this
                    # allows the lines to be reordered based on LAMMPS_ID. This
                    # is required as by default LAMMPS does not sort by ID, so
                    # the same atom will not appear in the same place for each
                    # time step.
                    lines = []
                    for _ in range(n_atoms):
                        line = file_handler.readline().split()
                        # convert id to int
                        line[i_id] = int(line[i_id])
                        lines.append(line)
                    # sort list of lists based on id
                    lines = sorted(lines, key=lambda x: x[i_id])

                    atoms = []
                    for line in lines:
                        # Checks if only specific atom_IDs are required, and if
                        # so, only creates atoms which have those IDs
                        if not atom_IDs or line[i_id] in atom_IDs:
                            atoms.append(create_atom(line))

                    # Multiply the number of timesteps by dt to calculate the
                    # elapsed time
                    configs.append(TemporalConfiguration(frame * time_step,
                                                         *atoms))

                    # next_frame_n next attribute is assigned dynamically
                    #pylint: disable=no-member
                    next_frame_n = next(frame_indexes)
                frame_n += 1
                if stop is not None and frame_n >= stop:
                    break

            line = file_handler.readline()
    return Trajectory(*configs)

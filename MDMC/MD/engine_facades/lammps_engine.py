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
from itertools import count, product, tee
from tempfile import NamedTemporaryFile

from lammps import PyLammps

from MDMC.common import units
from MDMC.MD.engine_facades.facade import MDEngine
from MDMC.MD.structural_units import BondedInteraction


class LAMMPSEngine(MDEngine):

    """
    Facade for LAMMPS

    Attributes:
    atom_dict - a dictionary with {MDMC_atom: LAMMPS_atom}, where MDMC_atom is
    an MDMC Atom object and LAMMPS_atom is the corresponding LAMMPS Atom object
    atom_types - a dictionary with {type_ID: MDMC_atom_group}, where the type_ID
    is a unique integer and MDMC_atom_group is a list of atoms which are
    identical in terms of element and interactions
    """

    @property
    def saved_config(self):

        raise NotImplementedError

    def setup_universe(self, universe, **settings):

        """
        Potential order of commands for setting up a LAMMPS universe:

        units(=real)
        atom_style (default = full)

        create_atoms

        non-bonded interactions
        bonded interactions
        """

        self.universe = universe

        # Create a PyLammps wrapper to capture LAMMPS output
        self.lmp = PyLammps()

        self.lmp.units('real')
        self.lmp.atom_style(settings.get('atom_style', 'full'))

        self.atom_dict = {}
        self.atom_types = {}
        self.atom_type_properties = []

        self._define_simulation_box(universe)
        self._build_configuration(universe)
        self._add_topology(universe)
        self.update_parameters(universe)
        raise NotImplementedError

    def setup_simulation(self, **settings):

        """
        Potential order of commands for setting up a LAMMPS simulation

        velocity

        neighbor
        neigh_modify

        timestep

        fix shake (or rattle)
        fix momentum
        """

        self._saved_config = None
        raise NotImplementedError

    def minimize(self, n_steps):

        """
        LAMMPS cannot minimize if constraints (SHAKE or RATTLE) are applied

        Potential order of commands for minimizing a LAMMPS simulation

        Remove fix shake or rattle if they exist
        """

        raise NotImplementedError

    def run(self, n_steps, equilibration):

        """
        Potential order of commands for running a LAMMPS simulation

        fix nve/nvt/npt
        fix temp/berendsen - if equilibrating with nve

        dump atom
        dump_modify sort

        run
        """

        self.trajectory_file = NamedTemporaryFile()
        self.lmp.dump('traj1', 'all', 'custom', traj_step,
                      self.trajectory_file.name, 'id', 'type', 'x', 'y', 'z')
        raise NotImplementedError

    def convert_trajectory(self):

        return convert_trajectory(self.trajectory_file.name,
                                  self.atom_type_properties,
                                  self.universe)

    def update_parameters(self):

        raise NotImplementedError
        # self._update_charges()
        # self._update_bonds()
        # self._update_angles()
        # self._update_dispersion()

    def save_config(self):

        raise NotImplementedError

    def reset_config(self):

        raise NotImplementedError

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
        bonded_interaction_types = [i.name for i in universe.interactions
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
        self.atom_type_properties = [(atom[0].element, atom[0].mass) for atom
                                     in sorted(self.atom_types.values())]

        for type_ID, atom_type_group in self.atom_types.items():
            self.lmp.mass(type_ID, atom_type_group[0].mass)
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

        return max([len(filter(lambda i: i.name == 'Bond', atom.interactions))
                    for atom in atoms])

    def _add_topology(self, universe):

        IMPERR = ('This interaction type has not been implemented in the LAMMPS'
                  ' facade')

        bonds, angles, disps, couls, others = partition_interactions(
            set(universe.interactions),
            ['Bond', 'BondAngle', 'Dispersion', 'Coulombic'],
            unpartitioned=True,
            lst=True)

        if others:
            raise NotImplementedError('Only bond, angle, dispersion and'
                                      ' coulombic interactions are implemented'
                                      ' in LAMMPS facade')

        # LAMMPS uses pair_style for all nonbonded interactions, so dispersive
        # and coulombic interactions are treated together. While multiple
        # identical pair_styles can be used with the hybrid command, it is
        # inefficient, so duplicates are removed with set.
        nonbonded_styles = set([parse_nonbonded_styles(nb) for nb
                                in disps + couls])
        if nonbonded_styles:
            self.lmp.pair_style('hybrid', *nonbonded_styles)
            self._create_Coulombic()
            self._update_charges()
            self._update_dispersions()

        if bonds:
            self.lmp.bond_style('hybrid',
                                *[parse_bonded_styles(b) for b in bonds])
            self._update_bonds(bonds)

        if angles:
            self.lmp.angle_style('hybrid',
                                 *[parse_bonded_styles(a) for a in angles])
            self._update_angles(angles)

    def _create_Coulombic(self, couls):

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
        for coul in couls:
            for atom_type in coul.atom_types:
                self.lmp.pair_coeff(atom_type, '*',
                                    parse_nonbonded_styles(coul))

    def _update_charges(self):

        """
        Updates the charges in LAMMPS
        """

        for atom, L_atom in self.atom_dict.items():
            self.lmp.set('atom',
                         L_atom.id,
                         convert_unit(atom.charge, atom.charge.unit))

    def _update_dispersions(self, disps):

        """
        Updates dispersion interactions in LAMMPS

        Arguments:
        disps - a list of dispersion interactions
        """

        for disp in disps:
            atom_type_pairs = product(disps.atom_types[0], disps.atom_types[1])
            for atom_type_pair in atom_type_pairs:
                self.lmp.pair_coeff(atom_type_pair[0], atom_type_pair[1],
                                    parse_dispersion_coefficients(disp))

    def _update_bonds(self, bonds, coeffs=False):

        """
        Updates bonds in LAMMPS

        Arguments:
        bonds - a list of bonds
        coeffs - a boolean specifying if the bond_coeffs are created
        """

        special = 'no'
        for ID, bond in enumerate(bonds, start=1):
            if coeffs:
                self.lmp.bond_coeff(ID, *parse_bond_coefficients(bond))

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

    def _update_angles(self, atoms):

        raise NotImplementedError


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


def convert_unit(value, unit):

    """
    Converts between MDMC units and LAMMPS real units

    Arguments:
    value - a float specifying the value in MDMC units
    unit - the unit of the value

    Returns:
    a float with the value in LAMMPS units
    """

    # As values must be unique in MDMC system of units dictionary
    # (units.SYSTEM), the keys and values can be inverted
    SYSTEM_INV = {unit:property for property, unit in units.SYSTEM.items()}

    # Apply conversion factor from units module based on LAMMPS unit of property
    # First try based on units module having exact unit
    try:
        value *= getattr(units, SYSTEM[SYSTEM_INV[unit]])
    except (KeyError, AttributeError):
        # Then try each component in turn
        for component in unit.components['numerator']:
            value *= getattr(units, SYSTEM[SYSTEM_INV[component]])
        for component in unit.components['denominator']:
            value /= getattr(units, SYSTEM[SYSTEM_INV[component]])

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
    a string with the correspoding LAMMPS pair style
    """

    lmp_str = []
    if interaction.function_name == 'LennardJones':
        lmp_str.append('lj')
    elif interaction.function_name == 'Coulomb':
        lmp_str.append('coul')
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    if interaction.cutoff:
        cutoff = convert_unit(interaction.cutoff, interaction.cutoff.unit)
        if interaction.kspace_solver:
            lmp_str.append('long')
        else:
            lmp_str.append('cut')
        lmp_str.append(str(cutoff))
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')

    return lmp_str


def parse_bond_coefficients(interaction):

    """
    Orders MDMC Parameters for input to LAMMPS bond_coeff

    Arguments:
    style - a string specifying the MDMC InteractionFunction name
    parameters - a NumPy array of the parameters, as is stored in
    InteractionFunction.params

    Returns:
    A list of style and parameters converted to the input format for LAMMPS
    bond_coeff
    """

    parameters = {p.name:convert_unit(p.value, p.unit)
                  for p in interaction.params}
    style = parse_bonded_styles(interaction)

    if style == 'harmonic':
        ordered_parameters = [parameters['potential_strength'],
                              parameters['equilibrium_state']]

    return [style] + ordered_parameters


def parse_dispersion_coefficients(interaction):

    """
    Orders MDMC Parameters for input to LAMMPS pair_coeff

    Arguments:
    interaction - an MDMC interaction object

    Returns:
    A list of style and parameters converted to the input format for LAMMPS
    pair_coeff
    """

    parameters = {p.name:convert_unit(p.value, p.unit)
                  for p in interaction.params}
    style = parse_nonbonded_styles(interaction)

    if style == 'lj':
        ordered_parameters = [parameters['epsilon'],
                              parameters['sigma']]

    return [style] + ordered_parameters


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

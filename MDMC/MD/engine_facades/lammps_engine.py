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
from itertools import product, tee

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
        Potential order of commands for runnibg a LAMMPS simulation

        fix nve/nvt/npt
        fix temp/berendsen - if equilibrating with nve

        dump atom
        dump_modify sort

        run
        """

        raise NotImplementedError

    def convert_trajectory(self):

        raise NotImplementedError

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

        self.atom_types = self._assign_atom_types(universe.atom_list)

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

    def _assign_atom_types(self, atoms):

        """
        Groups the atoms by element and interactions

        Arguments:
        atoms - a list of atoms

        Returns:
        dict with a key of ID (unique number) and a value of a list of all
        atoms that have the same element and interactions
        """

        atom_types_interactions = defaultdict(list)
        for atom in atoms:
            key = (atom.element, ) + tuple(sorted(atom.interactions))
            atom_types_interactions[key].append(atom)

        type_ID = 1
        atom_types = {}
        for atom_type_group in atom_types_interactions.values():
            atom_types[type_ID] = atom_type_group
            type_ID += 1

        return atom_types

    def _add_topology(self, universe):

        IMPERR = ('This interaction type has not been implemented in the LAMMPS'
                  ' facade')

        # Coulombic interactions are disregarded as these are set in
        # self._update_charges
        bonds, angles, disps, _, others = partition_interactions(
            set(universe.interactions),
            ['Bond', 'BondAngle', 'Dispersion', 'Coulombic'],
            unpartitioned=True,
            lst=True)

        if others:
            raise NotImplementedError('Only bond, angle, dispersion and'
                                      ' coulombic interactions are implemented'
                                      ' in LAMMPS facade')

        if disps:
            self.lmp.pair_style('hybrid',
                                *)


        if bonds:
            self.lmp.bond_style('hybrid',
                                *[parse_bonded_styles(b.function_name)
                                  for b in bonds])
            self._update_bonds(bonds)

        if angles:
            self.lmp.angle_style('hybrid',
                                 *[parse_bonded_styles(a.function_name)
                                   for a in angles])
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
                         convert_units(atom.charge, atom.charge.unit))

    def _update_dispersion(self, bonds):

        """
        Updates dispersion interactions in LAMMPS

        # TODO: Consider if it's useful to start by explictly setting all interaction pairs to 0
        raise NotImplementedError
        Arguments:
        disps - a list of dispersion interactions
        """

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
                self.lmp.bond_coeff(ID, *parse_bond_coefficients(
                    parse_bonded_styles(bond.function_name), bond.parameters))

            # Special triggers the internal interaction list in LAMMPS
            # This must at least occur at the end, and is an expensive
            # operation
            if bond is bonds[-1]:
                special = 'yes'
            for atom_tpl in bond.atoms:
                atom_IDs = [self.atom_dict[atom].id for atom in atom_tpl]
                self.lmp.create_bonds('single/bond',
                                      ID,
                                      *atom_IDs,
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
    'MASS':units.Unit('g / mol'),
    'CHARGE':units.Unit('e'),
    'ANGLE':units.Unit('deg'),
    'TEMPERATURE':units.Unit('K'),
    'ENERGY':units.Unit('kcal / mol'),
    'FORCE':units.Unit('kcal / mol Ang'),
    'PRESSURE':units.Unit('atm')
}

def convert_units(value, unit):

    """
    Converts between MDMC units and LAMMPS real units

    Arguments:
    value - a float specifying the value in MDMC units
    unit - the unit of the value

    Returns:
    a float with the value in LAMMPS units
    """

    raise NotImplementedError


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
        cutoff = convert_units(interaction.cutoff, interaction.cutoff.units)
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

    parameters = {p.name:convert_units(p.value, p.unit)
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

    parameters = {p.name:convert_units(p.value, p.unit)
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

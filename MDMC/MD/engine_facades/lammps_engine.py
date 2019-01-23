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


from itertools import tee

from lammps import PyLammps

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

        self._define_simulation_box(universe)
        self.atom_dict = {}
        self.atom_types = {}

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
        self._update_charges(self.universe)
        self._update_bonds(self.universe)
        self._update_angles(self.universe)

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
        self.lmp.create_box(n_elements,
                            region_ID,
                            nbondtypes=n_bond_types,
                            nangletypes=n_angle_types,
                            ndihedraltypes=n_dihedral_types,
                            nimpropertypes=n_improper_types)
    def _update_charges(self):

        for atom, L_atom in self.atom_dict.items():
            self.lmp.set('atom',
                         L_atom.id,
                         convert_units(atom.charge, atom.charge.unit))

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


def convert_units(value, unit):

    """
    Converts between MDMC units and LAMMPS real units

    Arguments:
    value - a float specifying the value in MDMC units
    unit - the unit of the value
    """

    raise NotImplementedError


def parse_bonded_styles(style):

    """
    Converts MDMC InteractionFunction names for BondedInteractions to LAMMPS
    bond styles

    Arguments:
    style - a string specifying the MDMC InteractionFunction name

    Returns:
    a string with the corresponding LAMMPS bond style
    """

    if style == 'HarmonicPotential':
        return 'harmonic'
    else:
        raise NotImplementedError('This InteractionFunction has not been'
                                  ' implemented in the LAMMPS facade')


def parse_bond_coefficients(style, parameters):

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

    parameters = {p.name:p.value for p in parameters}

    if style == 'harmonic':
        ordered_parameters = [parameters['potential_strength'],
                              parameters['equilibrium_state']]

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

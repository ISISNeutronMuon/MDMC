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

        # Create a PyLammps wrapper to capture LAMMPS output
        self.lmp = PyLammps()

        self.lmp.units('real')
        self.lmp.atom_style(settings.get('atom_style', 'full'))

        self._define_simulation_box(universe)

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

        # Remove when implementation complete
        raise NotImplementedError


def convert_units(self):

    """
    Converts between MDMC units and LAMMPS real units

    Arguments:
    value - a float specifying the value in MDMC units
    unit - the unit of the value
    """

    raise NotImplementedError

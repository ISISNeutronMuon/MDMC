"""Module for interfacing to ASE configuration readers (such as cif and pdb)

It should be possible to extend the CIF reader to include the additional
definitions in the mmCIF format.
"""

from itertools import groupby

from ase.io.cif import read_cif
import numpy as np

from MDMC.MD.ase.convert_atoms import ASEAtoms, convert_from_ase_atom
from MDMC.MD.structural_units import Bond, BondAngle, Coulombic, DihedralAngle
from MDMC.MD.interaction_functions import Coulomb



def ase_read_cif(file, index=0, add_bonds=True, add_charges = True, **settings):

    """
    Reads a configuration file and returns an MDMC Configuration

    If `names` or `atom_types` is passed, then equivalent interactions
    (`Coulombic` and `BondedInteraction`, if bonded interactions are defined in
    the CIF file) will be initialized as a single object. For instance if the
    CIF file includes a benzene ring, then as long as the correct `names` or
    `atom_types` are passed, then there will only be a single C-C `Bond` object,
    which will include all 6 of the atom pairs.
    If both `names` and `atom_types` are passed, `atom_types` will be used to
    group `Atom` objects.
    If neither `names` or `atom_types` is passed then each interaction in will
    become a separate object.

    .. note:: Not all CIF files contain bonded interactions (it is only common
    for biomolecules).

    .. note:: improper dihedrals are not explicitly defined in CIF, so these
    must be set after initialization of `DihedralAngle` objects.

    Parameters
    ----------
    file : File, str
        A `File` object, or the absolute file name of the configuration file
    index : int, optional
        The index of the configuration in the CIF file. Only a single
        configuration can be read from a CIF file, with the default being the
        first (index=0) configuration.
    add_bonds : bool, optional
        Whether or not any bonded interactions defined in the CIF file will be
        included. By default this is True.
    add_charges : bool, optional
        Whether or not each atom in the CIF file will be assigned a `Coulombic`
        interaction with a `Coulomb` function. CIF files do not contain charge
        information, so the charge of the `Coulombic` interaction will be set to
        0. This enables the charges to be set by the application of a
        `ForceField` object. By default this is True.
    **settings
        names : list of str
            A list of names for the atoms in the CIF file. These names must have
            the same order as the order the atoms in the file. A `name` must be
            be provided for each atom in the CIF file.
        atom_types : list of int
            A list of int for atom types of the atoms in the CIF file. An
            `atom_type` must be provided for each atom in the CIF file.
        cutoff : float
            A distance (in Ang) at which the `Coulombic` interactions are
            cutoff. If this is not passed, the `cutoff` will be set to 10.

    Returns
    -------
    list of Atom
        The `Atom` objects corresponding to the data in the CIF file
    """

    # ASE does not explicity define bonds, however when it reads a CIF file, it
    # also reads the bonding information (if this is defined in the file).
    # This information is not used by ASE, however it is included in the info
    # attribute of any Atoms objects if the store_tags parameter is set to True
    images = read_cif(file, index=slice(index, None, None), store_tags=True)

    # images is a generator with a single element, an ase.atoms.Atoms object
    ase_atoms = list(images)[0]

    ase_atoms = _reduce_ase_unit_cell(ase_atoms)

    # Use provided names or None for each Atom
    names = settings.get('names', [None] * len(ase_atoms))
    atom_types = settings.get('atom_types', [None] * len(ase_atoms))

    # dict of CIF atom label to MDMC atom
    atoms_labels = {label:convert_from_ase_atom(atom,
                                                name=name,
                                                atom_type=atom_type,
                                                set_charge=False)
                    for label, atom, name, atom_type
                    in zip(ase_atoms.info['_atom_site_label'],
                           ase_atoms,
                           names,
                           atom_types)}
    atoms = list(atoms_labels.values())

    # The keys are used to group atoms so that the same atoms (atom tuples) are
    # used for a single Coulombic (BondedInteraction). So if atom_types are
    # used, a single Coulombic interaction will be created for all atoms with
    # atom_type 1.
    if atom_types[0]:
        coulombic_key = lambda atom: atom.atom_type
        bonded_key = lambda atom_arr: [atom.atom_type for atom in atom_arr]
    elif names[0]:
        coulombic_key = lambda atom: atom.name
        bonded_key = lambda atom_arr: [atom.name for atom in atom_arr]
    else:
        coulombic_key = bonded_key = None

    _create_coulombic_interactions(atoms,
                                   settings.get('cutoff', 10.0),
                                   coulombic_key)

    if add_bonds:
        # The CIF defintions which relate to bonded interactions. Note that
        # _chemical_conn_bond is not included in tags which describe bonds, even
        # though it could be used to create bonds. This is because there is no
        # equivalent tag for angles and dihedrals. It also does not include H
        # bonds, as these are not currently defined in MDMC.
        cif_geom_defs = ['_geom_bond_atom_site_label_',
                         '_geom_angle_atom_site_label_',
                         '_geom_torsion_atom_site_label_']
        for cif_geom_def in cif_geom_defs:
            # For the CIF file to contain all the labels of the corresponding
            # cif_geom_def, it must contain at least the first label, '1'.
            if cif_geom_def + '1' in ase_atoms.info:
                inters_atoms = get_bonded_interactions_atoms(ase_atoms.info,
                                                             cif_geom_def,
                                                             atoms_labels)
                _create_bonded_interactions(inters_atoms, bonded_key)

    return atoms


def get_bonded_interactions_atoms(ase_atoms_info, cif_geom_def, atoms_labels):

    """
    Gets the atoms for each bonded interaction

    Parameters
    ----------
    ase_atoms_info : dct
        A dictionary containing
    cif_geom_def : str
    atoms_labels

    Returns
    -------
    np.array
        A 2D array with dimensions (n_interactions, n_atoms_per_interaction). So
        for 5 bond interactions, the dimensions of the array will be (5, 2),
        with the zeroeth index containing the two Atoms involved in the zeroeth
        bond, the first index containing the two Atoms involved in the first
        bond etc. For bond angle and dihedral interactions, the order of the
        atoms corresponds to the order required for BondAngle and DihedralAngle
        interactions.
    """

    # There are a maximum of 4 atom sites in a geometry definition (for
    # torsions)
    cif_geom_defs = [cif_geom_def + str(index) for index in range(1, 5)]

    site_labels = np.array([ase_atoms_info[geom_def] for geom_def
                            in cif_geom_defs if geom_def in ase_atoms_info]).T
    label_to_atom = np.vectorize(atoms_labels.__getitem__)
    interactions_atoms = label_to_atom(site_labels)

    return interactions_atoms


def _create_coulombic_interactions(atoms, cutoff, key=None):

    """
    Creates `Coulombic` interactions

    Parameters
    ----------
    atoms : list of Atom
        The `Atom` objects for which a `Coulombic` interaction will be created
    key : function
        The key which will be used to group `Atom` objects. Grouped `Atom`
        objects will have a single `Coulombic` interaction.
    """

    if key:
        atoms = _group_atoms(atoms, key)
    # If no grouping then each atom_group is a single atom
    for atom_group in atoms:
        # A Coulomb function is set so that
        Coulombic(atoms=atom_group, cutoff=cutoff, function=Coulomb(0.))


def _create_bonded_interactions(interactions_atoms, key=None, **settings):

    """
    Creates `BondedInteraction` objects

    Parameters
    ----------
    **settings
        Settings to be passed for BondedInteraction initialization. For example
        improper=True can be passed to initialize a DihedralAngle to be an
        improper dihedral.

    Raises
    ------
    TypeError
        If the number of atoms for each interaction is not a valid number to
        create a bonded interaction
    """

    # Determine bonded interaction type based on number of atoms it is between
    n_inter_atoms = np.shape(interactions_atoms)[1]
    if n_inter_atoms == 2:
        bond_type = Bond
    elif n_inter_atoms == 3:
        bond_type = BondAngle
    elif n_inter_atoms == 4:
        bond_type = DihedralAngle
    else:
        raise TypeError('{} is not a valid number of atoms for a bonded'
                        ' interaction'.format(n_inter_atoms))

    if key:
        interactions_atoms = _group_atoms(interactions_atoms, key)

    for interaction_atoms in interactions_atoms:
        # BondedInteractions require atoms to hashable, so tuple of np.arrays
        # must be mapped to tuple of tuples
        bond_type(*tuple(map(tuple, interaction_atoms)), **settings)


def _group_atoms(atoms, key):

    """
    Groups atoms based on a key

    Parameters
    ----------
    atoms : list of Atom
        The `Atom` objects to be grouped
    key : function
        The `key` with which the Atom objects will be grouped

    Returns
    -------
    list of tuples
        Where each tuple contains a group of equivalent `Atom` objects, based on
        `key`
    """

    atoms = sorted(atoms, key=key)
    return [tuple(group) for _, group in groupby(atoms, key=key)]


def _reduce_ase_unit_cell(ase_atoms):

    """
    Reduces an ase.atoms.Atoms object from a unit cell of molecules to a single
    molecule

    Uses the number of atoms present in the CIF file to determine the size of
    the molecule

    Parameters
    ----------
    ase_atoms : ase.atoms.Atoms
        An ase.atoms.Atoms object from which a single molecule will be extracted

    Returns
    -------
    ASEAtoms
        An ASeAtoms objects containing the atoms of a single molecule
    """

    # The number of atoms in a molecule can be determined from info in CIF file
    n_atoms_molecule = len(ase_atoms.info['_atom_site_label'])

    # If the ase_atoms object already contains correct number of atoms, return
    # it
    if len(ase_atoms) == n_atoms_molecule:
        return ase_atoms

    # Otherwise build a new atoms object from the CIF fractional atom positions,
    # which are also stored in ase_atoms.info
    positions = np.array([ase_atoms.info['_atom_site_fract_' + dim]
                          for dim in ['x', 'y', 'z']]).T
    formula = get_reduced_chemical_formula(ase_atoms.get_chemical_symbols(),
                                           n_atoms_molecule)
    return ASEAtoms(formula, scaled_positions=positions, cell=ase_atoms.cell,
                    info=ase_atoms.info)


def get_reduced_chemical_formula(symbols, n_atoms):

    """
    Parameters
    ----------
    symbols : list of str
        The chemical formula to be reduced. It is expressed as a list of
        elements, with a single element for each atom. Elements are grouped by
        type but not ordered e.g. all 'O' values, then all 'H' values etc.
    n_atoms : int
        The total number of atoms that will be in the reduced formula

    Returns
    -------
    str
        The chemical formula corresponding to symbols, except with only n_atoms

    Example
    -------
    Reducing the formula for three water molecules to a single water molecules::

        >>> get_reduced_chemical_formula(['H'] * 6 + ['O'] * 3, 3)
        'H2O'

    """

    return ''.join(symbols[::len(symbols) // n_atoms])

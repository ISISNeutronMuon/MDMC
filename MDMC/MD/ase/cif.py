"""Module for interfacing to ASE configuration readers (such as cif and pdb)

It should be possible to extend the CIF reader to include the additional
definitions in the mmCIF format.

It is expected that some of the behaviour in ``ase_read_cif`` will be extracted
into ``MDMC.readers.configurations.cif`` when an MDMC CIF reader is implemented.
"""

from itertools import groupby
from typing import TYPE_CHECKING, Callable

from ase.io.cif import read_cif
import numpy as np

from MDMC.MD.ase.conversions import ASEAtoms, convert_from_ase_atom
from MDMC.MD.structures import (
    BoundingBox, get_reduced_chemical_formula)
from MDMC.MD.interactions import Coulombic, Bond, BondAngle, DihedralAngle
from MDMC.MD.interaction_functions import Coulomb

if TYPE_CHECKING:
    from MDMC.MD.structures import Atom
    import ase


def ase_read_cif(file: str, **settings: dict) -> 'list[Atom]':
    """
    Reads a configuration file and returns a list of ``Atom`` objects.

    These Atom objects can optionally have ``Coulombic`` interactions and also
    ``BondedInteraction`` objects if bonded interactions are defined in the CIF
    file.

    If ``names`` or ``atom_types`` is passed, then equivalent interactions
    (``Coulombic`` and ``BondedInteraction``, if bonded interactions are defined
    in the CIF file) will be initialized as a single object. For instance if the
    CIF file includes a benzene ring, then as long as the correct ``names`` or
    ``atom_types`` are passed, then there will only be a single C-C ``Bond``
    object, which will include all 6 of the atom pairs.
    If both ``names`` and ``atom_types`` are passed, ``atom_types`` will be used
    to group ``Atom`` objects.
    If neither ``names`` or ``atom_types`` is passed then each interaction will
    become a separate object.

    .. note:: Not all CIF files contain bonded interactions (it is only common
              for biomolecules).

    .. note:: improper dihedrals are not explicitly defined in CIF, so these
              must be set after initialization of ``DihedralAngle`` objects.

    .. note:: CIF reader cannot parse CIF files with user defined text sections,
              so these must be stripped out before reading.

    Parameters
    ----------
    file : file, str
        A ``file``, or the absolute file name of the configuration file
    **settings
        ``index`` (`int`, optional)
            The index of the configuration in the CIF file. Only a single
            configuration can be read from a CIF file, with the default being
            the first (``index=0``) configuration.
        ``names`` : (`list` of `str`)
            A `list` of ``names`` for the atoms in the CIF file. These ``names``
            must have the same order as the order the atoms in the file. A
            ``name`` must be be provided for each atom in the CIF file.
        ``atom_types`` : (`list` of `int`)
            A `list` of `int` for atom types of the atoms in the CIF file. These
            names must have the same order as the order the atoms in the file.
            An ``atom_type`` must be provided for each atom in the CIF file.
        ``cutoff`` : (`float`)
            A distance (in Ang) at which the ``Coulombic`` interactions are
            cutoff. If this is not passed, the ``cutoff`` will be set to 10.
        ``add_bonds`` : (`bool`, optional)
            Whether or not any bonded interactions defined in the CIF file will
            be included. By default this is `True`.
        ``add_charges`` : (`bool`, optional)
            Whether or not each atom in the CIF file will be assigned a
            ``Coulombic`` interaction with a ``Coulomb`` function. CIF files do
            not contain charge information, so the charge of the ``Coulombic``
            interaction will be set to 0. This enables the charges to be set by
            the application of a ``ForceField`` object. By default this is True.

    Returns
    -------
    `list` of ``Atom``
        The ``Atom`` objects corresponding to the data in the CIF file
    """

    index = settings.get('index', 0)
    add_bonds = settings.get('add_bonds', True)
    add_charges = settings.get('add_charges', True)

    # ASE does not explicitly define bonds, however when it reads a CIF file, it
    # also reads the bonding information (if this is defined in the file).
    # This information is not used by ASE, however it is included in the info
    # attribute of any Atoms objects if the store_tags parameter is set to True
    images = read_cif(file, index=slice(index, None, None), store_tags=True)

    # images is a generator with a single element, an ase.atoms.Atoms object
    try:
        ase_atoms = list(images)[0]
    except Exception as error:
        msg = ('MDMC uses the ASE (Atomic Simulation Environment) module '
               f'for reading your CIF file ({file.name}), '
               'which failed. Please see the ASE CIF documentation for help '
               'with the CIF format ASE requires.'
               ' Please note that the ASE CIF reader cannot parse CIF files '
               'with user defined text sections so these '
               'must be stripped out before reading. '
               'The full Python stack error message should be shown above.')
        raise Exception(msg).with_traceback(error.__traceback__)

    ase_atoms = _reduce_ase_unit_cell(ase_atoms)

    # Use provided names or None for each Atom
    names = settings.get('names', [None] * len(ase_atoms))
    atom_types = settings.get('atom_types', [None] * len(ase_atoms))

    # dict of CIF atom label to MDMC atom
    atoms_labels = {label: convert_from_ase_atom(atom,
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
        def coulombic_key(atom):
            return atom.atom_type
        def bonded_key(atom_arr):
            return [atom.atom_type for atom in atom_arr]
    elif names[0]:
        def coulombic_key(atom):
            return atom.name
        def bonded_key(atom_arr):
            return [atom.name for atom in atom_arr]
    else:
        coulombic_key = bonded_key = None

    if add_charges:
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

    _make_atom_positions_valid(atoms)

    return atoms


def get_bonded_interactions_atoms(ase_atoms_info: dict,
                                  cif_geom_def: str,
                                  atoms_labels: dict) -> np.ndarray:
    """
    Gets the atoms for each bonded interaction

    Parameters
    ----------
    ase_atoms_info : dict
        A `dict` containing site labels for one or more of bonds, angles, and
        torsions. The corresponding values are a list with label (str) for each
        interaction. The number of site label keys is 2 for bonds, 3 for angles
        and 4 for torsions. For instance, for bonds there should be
        '_geom_bond_atom_site_label_1' and '_geom_bond_atom_site_label_2', with
        each being a list containing the first (or second) site label for each
        interaction.
    cif_geom_def : str
        Specifies whether the interaction type is a bond, angle, or torsion
    atoms_labels : dict
        (label:atom) pairs, where label is a str with the _atom_site_label from
        the CIF file, and atom is the corresponding MDMC ``Atom`` object.

    Returns
    -------
    numpy.ndarray
        A 2D ``array`` with dimensions (n_interactions,
        n_atoms_per_interaction). So for 5 bond interactions, the dimensions of
        the array will be (5, 2), with the zeroeth index containing the two
        ``Atoms`` involved in the zeroeth bond, the first index containing the
        two ``Atoms`` involved in the first bond etc. For bond angle and
        dihedral interactions, the order of the atoms corresponds to the order
        required for ``BondAngle`` and ``DihedralAngle`` interactions.
    """

    # There are a maximum of 4 atom sites in a geometry definition (for
    # torsions)
    cif_geom_defs = [cif_geom_def + str(index) for index in range(1, 5)]
    site_labels = np.array([ase_atoms_info[geom_def] for geom_def
                            in cif_geom_defs if geom_def in ase_atoms_info]).T
    label_to_atom = np.vectorize(atoms_labels.__getitem__)
    interactions_atoms = label_to_atom(site_labels)

    return interactions_atoms


def _create_coulombic_interactions(atoms: 'list[Atom]', cutoff: float,
                                   key: Callable = None) -> None:
    """
    Creates ``Coulombic`` interactions

    Parameters
    ----------
    atoms : list of Atom
        The ``Atom`` objects for which a ``Coulombic`` interaction will be
        created
    cutoff: float
        The cutoff distance for the atom's Coulombic interaction.
    key : function
        The key which will be used to group ``Atom`` objects. Grouped ``Atom``
        objects will have a single ``Coulombic`` interaction.
    """

    atoms = _group_atoms(atoms, key)
    # If no grouping then each atom_group is a single atom
    for atom_group in atoms:
        # A Coulomb function is set so that the interaction can be parametrized
        Coulombic(atoms=atom_group, cutoff=cutoff, function=Coulomb(0.))


def _create_bonded_interactions(interactions_atoms: np.ndarray,
                                key: Callable = None,
                                **settings: dict) -> None:
    """
    Creates ``BondedInteraction`` objects

    Parameters
    ----------
    interaction_atoms : numpy.ndarray
        A 2D ``array`` with dimensions (n_interactions,
        n_atoms_per_interaction). So for 5 bond interactions, the dimensions of
        the array must be (5, 2), with the zeroeth index containing the two
        ``Atoms`` involved in the zeroeth bond, the first index containing the
        two ``Atoms`` involved in the first bond etc. For bond angle and
        dihedral interactions, the order of the atoms must correspond to the
        order required for ``BondAngle`` and ``DihedralAngle`` interactions.
    key : function
        A function by which the atoms tuples are grouped. Each group will have a
        single bonded interaction applied. For example, for a bonded interaction
        where the key is atom_type, if an atom pair has ``atom_types`` (1, 2),
        this will be grouped with all other atom pairs with ``atom_types``
        (1, 2), and a single ``Bond`` will be applied to the group.
    **settings
        Settings to be passed for ``BondedInteraction`` initialization. For
        example ``improper=True`` can be passed to initialize a
        ``DihedralAngle`` to be an improper dihedral.

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
        raise TypeError(f'{n_inter_atoms} is not a valid number of atoms for a bonded'
                        ' interaction')

    # If no key is passed, group by id - this will result in each tuple of atoms
    # being in its own group (i.e. no tuples of atoms are grouped together).
    # This ensures that interactions_atoms will always have the correct
    # dimensions
    if key is None:
        key = id
    interactions_atoms = _group_atoms(interactions_atoms, key)

    for interaction_atoms in interactions_atoms:
        # BondedInteractions require atoms to hashable, so tuple of np.arrays
        # must be mapped to tuple of tuples
        bond_type(*tuple(map(tuple, interaction_atoms)), **settings)


def _group_atoms(atoms: 'list[Atom]', key: Callable) -> 'list[tuple[Atom]]':
    """
    Groups atoms based on a key

    Parameters
    ----------
    atoms : list of Atom
        The ``Atom`` objects to be grouped
    key : function
        The ``key`` with which the ``Atom`` objects will be grouped

    Returns
    -------
    list of tuples
        Where each tuple contains a group of equivalent ``Atom`` objects, based
        on ``key``
    """
    try:
        atoms = sorted(atoms, key=key)
    except TypeError:
        pass  # If the key passed in is None then it has no idea how to compare `Atom`s

    return [tuple(group) for _, group in groupby(atoms, key=key)]


def _reduce_ase_unit_cell(ase_atoms: 'ase.atoms.Atoms') -> ASEAtoms:
    """
    Reduces an ``ase.atoms.Atoms`` object from a unit cell of molecules to a
    single molecule

    Uses the number of atoms present in the CIF file to determine the size of
    the molecule

    Parameters
    ----------
    ase_atoms : ase.atoms.Atoms
        An ``ase.atoms.Atoms`` object from which a single molecule will be
        extracted

    Returns
    -------
    ASEAtoms
        An ``ASEAtoms`` object containing the atoms of a single molecule
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
                                           len(ase_atoms) // n_atoms_molecule,
                                           system=None)
    return ASEAtoms(formula, scaled_positions=positions, cell=ase_atoms.cell,
                    info=ase_atoms.info)


def _make_atom_positions_valid(atoms: 'list[Atom]') -> None:
    """
    Sets the positions of all atoms are positive (including 0.)

    This is so that all positions are valid within ``Universe``

    Parameters
    ----------
    atoms : list
        A `list` of `Atom` which will have their positions set so that relative
        distances are preserved and the smallest positions are equal to 0.
    """

    # Offset atom positions so that they are >= 0.
    box = BoundingBox(atoms)
    for atom in atoms:
        atom.position -= box.min

"""Module for interfacing to ASE configuration readers (such as cif and pdb)
"""

from ase.io.cif import read_cif
import numpy as np

from MDMC.MD.structural_units import Atom, Molecule, Bond, BondAngle, \
    DihedralAngle, Dispersion
from MDMC.MD.ase.convert_atoms import ASEAtoms, convert_from_ase_atom
from MDMC.trajectory_analysis.trajectory import Configuration


def ase_read_cif(file, index=0, **settings):

    """
    Reads a configuration file and returns an MDMC Configuration

    Parameters
    ----------
    file : File, str
        A File object, or the absolute file name of the configuration file
    index : int, optional
        The index of the configuration in the CIF file. Only a single
        configuration can be read from a CIF file, with the default being the
        first (index=0) configuration.
    **settings
        names : list of str
            A list of names for the atoms in the CIF file. These names must have
            the same order as the order the atoms in the file. A name must be
            be provided for each atom in the CIF file.

    Returns
    -------
    Configuration
        The Configuration corresponding to the data in file
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

    # dict of CIF atom label to MDMC atom
    atoms_labels = {label:convert_from_ase_atom(atom, name=name)
                    for label, atom, name
                    in zip(ase_atoms.info['_atom_site_label'], ase_atoms, names)
                   }

    # The CIF defintions which relate to bonded interactions
    cif_geom_defs = ['_geom_bond_atom_site_label_',
                     '_geom_angle_atom_site_label_',
                     '_geom_torsion_atom_site_label_']
    for cif_geom_def in cif_geom_defs:
        interactions_atoms = get_bonded_interactions_atoms(ase_atoms.info,
                                                           cif_geom_def)
        _create_bonded_interactions(atoms_labels, interactions_atoms)

    atoms = list(atoms_labels.values())
    return atoms


def get_bonded_interactions_atoms(ase_atoms_info, cif_geom_def):

    """
    ase_atoms_info : dct
    cif_geom_def : str
    """

    # There are a maximum of 4 atom sites in a geometry definition (for
    # torsions)
    cif_geom_defs = [cif_geom_def + str(index) for index in range(1, 5)]
    # Use indexes as keys
    return {int(geom_def[-1]): ase_atoms_info[geom_def] for geom_def
            in cif_geom_defs if geom_def in ase_atoms_info}


def _create_bonded_interactions(atoms_labels, interactions_atoms):

    # Determine bonded interaction type based on number of atoms it is between
    n_inter_atoms = len(interactions_atoms)
    if n_inter_atoms == 2:
        bond_type = Bond
    elif n_inter_atoms == 3:
        bond_type = BondAngle
    elif n_inter_atoms == 4:
        bond_type = DihedralAngle
    else:
        raise TypeError('{} is not a valid number of atoms for a bonded'
                        ' interaction'.format(n_inter_atoms))

    # Interaction atoms contains lists for each atom site label e.g. for a bond
    # angle it contains a three lists, where index 0 of each list are the labels
    # of the three atoms which comprise the first bond angle.
    for interaction_atoms in zip(*[interactions_atoms[i] for i
                                   in range(1, n_inter_atoms + 1)]):
        # Convert from label to atom and create the bonded interaction
        bond_type(*map(atoms_labels.__getitem__, interaction_atoms))


def _reduce_ase_unit_cell(ase_atoms):

    """
    Reduces an ase.atoms.Atoms object from a unit cell of molecules to a single
    molecule
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

    return ''.join(symbols[::len(symbols) // n_atoms])

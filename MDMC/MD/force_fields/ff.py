"""A module for defining force fields that can be applied to a universe

Each force field consists of a combination of interaction functions, and also
the values of the parameters within these functions.  In this instance water
models (such as SPCE and TIP3P) are also defined as force fields, even though
the parameter sets are restricted to describing water.  Each force field module
is self contained, although adding a new force field may require changes to the
MD engine facades, so that a correspondence is established between the MDMC
force field and the MD engine equivalent."""

from abc import ABC, abstractmethod
from inspect import signature
from itertools import chain, permutations
import logging
import os
from re import escape, sub

import pandas as pd

from MDMC.common.decorators import repr_decorator, weakref_cache
from MDMC.common.df_operations import filter_dataframe, filter_ordered_dataframe
from MDMC.MD.interactions import Coulombic, BondedInteraction
from MDMC.MD import interaction_functions


LOGGER = logging.getLogger(__name__)


@repr_decorator('interaction_dictionary')
class ForceField(ABC):

    """
    Abstract class defining a force field

    For each interaction type that it uses (non-bonded, bonds, bond angles etc),
    a force field must define the interaction function (LJ, harmonic etc).  It
    must also define the parameters for each of these functions.
    """

    @property
    @abstractmethod
    def interaction_dictionary(self):
        """
        The `dict` of interactions that exist within the ``ForceField``

        Returns
        -------
        dict
            {``Interaction``:``Elements``} where ``Elements`` is an ordered
            `tuple` of elemental symbols, and values of ``InteractionFunction``
            objects.

        """

        raise NotImplementedError

    def parameterize_interactions(self, interactions):
        """
        Parameterizes the interactions with the parameters speicifed in the
        ``interaction`` `dict`

        Parameters
        ----------
        interactions : list
            A `list` of ``Interaction`` objects to be parameterized
        """

        for interaction in interactions:
            self._parameterize_interaction(interaction)

    def _parameterize_interaction(self, interaction):
        """
        Parameterizes the interaction with the parameters specified in the
        ``interaction`` `dict`

        Parameters
        ----------
        interaction : Interaction
            An ``Interaction`` to be parameterized
        """

        elements = interaction.element_tuple()
        try:
            interaction.function = self.interaction_dictionary[
                (type(interaction), elements)]
            interaction.function.set_parameters_interactions(interaction)
        except KeyError as error:
            raise KeyError("This force field does not have defined interactions"
                           " for these element types") from error


@repr_decorator('interaction_dictionary', 'n_body')
class WaterModel(ForceField):

    """
    Abstract class for force fields that describe a water model
    """

    @property
    @abstractmethod
    def n_body(self):
        """
        This is the number of bodies in the water model.

        .. note:: THIS MUST BE IMPLEMENTED USING A STATIC VARIABLE, INSTEAD OF A
                  PROPERTY.

        Returns
        -------
        int
            The number of bodies (atoms) in the water model.
        """

        raise NotImplementedError


@repr_decorator('file_name')
class FileForceField(ForceField):

    """
    Abstract class for force fields that are read from files
    """

    def __init__(self):

        self.data = {}
        self._interaction_dictionary = None
        with open(self.absolute_path, encoding='UTF-8') as file:
            n_datatypes = self._parse_header(file.readline(), int)
            self.inter_functions = dict(self._parse_header(file.readline(),
                                                           str))
            line = file.readline()
            while 'Atoms' not in line:
                line = file.readline()
            for name, num in n_datatypes:
                # +1 rows to account for header
                datatype = pd.read_csv(file, nrows=num + 1, engine='python',
                                       sep='\t', skiprows=(1 if name != 'atoms'
                                                           else 0), dtype='str')
                # Convert columns to correct types
                for col_name in datatype:
                    datatype[col_name] = self._set_col_type(datatype[col_name])
                self.data[name] = datatype

        self.atom_name_group = dict(zip(zip(self.atoms.name,
                                            self.atoms.element),
                                        self.atoms.atom_group))
        self.atom_type_name = dict(zip(self.atoms.atom_type,
                                       self.atoms.name))

    @property
    def absolute_path(self):
        """
        Get the absolute path of the data

        Returns
        -------
        str
            The absolute path (including file name) of the force field data file
        """

        # Either the file name is defined with the path, in which case it is the
        # same as the absolute path
        if os.path.isfile(self.file_name):
            return self.file_name
        # Or it is assumed that the path is the force field data directory
        return (os.path.dirname(os.path.realpath(__file__))
                + '/data/'
                + self.file_name)

    @property
    @abstractmethod
    def file_name(self):
        """
        Get the file name of the data
        """

        raise NotImplementedError

    @property
    def atoms(self):
        """
        Get file parameters for the atoms defined within the force field

        May includes the normal number of bonds that this atom type should
        possess, which is not currently used but could be an additional check on
        the validity of the topology

        Returns
        -------
        pandas.DataFrame
            A ``DataFrame`` with the force field atom type, atom group, element,
            mass, charge, and name
        """

        return self.data['atoms']

    @property
    def bonds(self):
        """
        Get file parameters for the bonds of the force field

        Returns
        -------
        pandas.DataFrame
            A ``DataFrame`` with all bonds, atom groups, and the parameters of
            the bonds
        """

        return self.data['bonds']

    @property
    def bond_angles(self):
        """
        Get file parameters for the bond angles of the force field

        Returns
        -------
        pandas.DataFrame
            A ``DataFrame`` with all bond angles, atom groups, and the
            parameters of the bond angle
        """

        return self.data['bondangles']

    @property
    def propers(self):
        """
        Get file parameters for the proper dihedrals of the force field

        Returns
        -------
        pandas.DataFrame
            A ``DataFrame`` with all proper dihedrals, atom groups, and the
            parameters of the proper dihedral
        """

        return self.data['propers']

    @property
    def impropers(self):
        """
        Get file parameters for the improper dihedrals of the force field

        Returns
        -------
        pandas.DataFrame
            A ``DataFrame`` with all improper dihedrals, atom groups, and the
            parameters of the improper dihedral
        """

        return self.data['impropers']

    @property
    def dispersions(self):
        """
        Get file parameters for the dispersion interactions of the force field

        Returns
        -------
        pandas.DataFrame
            A ``DataFrame`` with all dispersion interactions, atom types, and
            the parameters of the dispersion interaction
        """

        return self.data['dispersions']

    @property
    def interaction_dictionary(self):

        # Cache interaction dictionary
        try:
            return self._interaction_dictionary
        except AttributeError:
            self._interaction_dictionary = None
            return self._interaction_dictionary

    def filter_element(self, element):
        """
        Filters the atoms in the ``FileForceField`` by element

        Parameters
        ----------
        element : str
            The element by which the atoms in the FileForceField will be
            filtered

        Returns
        -------
        pandas.DataFrame
            A filtered ``DataFrame`` where each row is an atom type that has an
            element specified by ``element``
        """

        return filter_dataframe([element], self.atoms, column_names=['element'])

    def set_atom_mass(self, atom):
        """
        Sets ``Atom.mass`` to the mass defined in the force field for that atom
        type

        Parameters
        ----------
        atom : Atom
            The ``Atom`` for which the mass will be set
        """

        ff_atom = filter_ordered_dataframe([atom.name, atom.element.symbol],
                                           self.atoms,
                                           column_names=['name', 'element'])
        atom.mass = ff_atom.mass

    def _parameterize_interaction(self, interaction):
        """
        Parameterizes the interaction with the parameters specified in the
        ``interaction`` `dict`

        Arguments:
        interaction : Interaction
            An ``Interaction`` to be parameterized
        """

        if isinstance(interaction, BondedInteraction):
            self._parametrize_bonded(interaction)
        elif isinstance(interaction, Coulombic):
            self._parametrize_coulombic(interaction)
        else:
            self._parametrize_dispersion(interaction)

    def _parametrize_bonded(self, bonded):
        """
        Parametrizes a ``BondedInteraction``

        Parameters
        ----------
        bonded : BondedInteraction
            A subclass of ``BondedInteraction`` which will be parametrized
        """

        groups = set()
        for atom_tuple in bonded.atoms:
            tuple_groups = []
            for atom in atom_tuple:
                self._convert_atom_type_name(atom)
                # Raise an error if the name/element combination can't be found
                try:
                    tuple_groups.append(self.atom_name_group[(atom.name,
                                                              atom.element.symbol)])
                except KeyError as error:
                    raise KeyError(f'Unable to find atom of element "{atom.element.symbol}" '
                                   f'recorded with the name "{atom.name}" '
                                   'in the specified force field file.') from error

            groups.add(tuple(tuple_groups))

        # Ensure that the groups of all atom tuples is consistent (i.e. each
        # atom tuple should contain the same groups, in the same order)
        # It is important to note that this is more restrictive than it needs to
        # be - it could allow not raise an error where the mismatch was valid
        # because wildcards would allow the interaction parameters to be the
        # same regardless of the different atom groups, however this is not
        # implemented.
        if len(groups) != 1:
            msg = (f'The atom groups of this interaction are not consistent {groups}.'
                   ' Atom tuples should have the same groups in the same'
                   ' order.')
            LOGGER.error('%s: %s',
                         self.__class__,
                         msg)
            raise ValueError(msg)

        groups = groups.pop()
        bonded_type = str(type(bonded).__name__)
        if bonded_type == 'DihedralAngle':
            bonded_type = 'improper' if bonded.improper else 'proper'
        function_type = self._get_interaction_function(bonded_type)

        parameter_df = getattr(self, sub('(?<!^)(?=[A-Z])',
                                         '_',
                                         bonded_type).lower() + 's')

        # Filter for both the specified order of groups, and the reverse
        # This is valid for all bonded types except improper dihedrals, which
        # for which only the first atom occupies a unique position (the center);
        # the other atoms can be in any permutation
        matches = []
        if bonded_type != 'improper':
            for ordered_groups in [groups, tuple(reversed(groups))]:
                regex = 'atom_group?'
                matches.append(filter_ordered_dataframe(ordered_groups,
                                                        parameter_df,
                                                        column_regex=regex,
                                                        wildcard=0))
        else:
            col_names = ['atom_group1']
            cen_atom_match = filter_ordered_dataframe([groups[0]],
                                                      parameter_df,
                                                      column_names=col_names,
                                                      wildcard=0)
            for permuted_groups in list(permutations(groups[1:])):
                # Only filter using atom_group2, atom_group3, atom_group4
                regex = '^atom_group[2-4]$'
                matches.append(filter_ordered_dataframe(permuted_groups,
                                                        cen_atom_match,
                                                        column_regex=regex,
                                                        wildcard=0))
        # Duplicate dropping requied for groups == tuple(reversed(groups))
        matches = pd.concat(matches).drop_duplicates()
        # If there is more than one row which matches the groups (because of
        # wildcards) then pick row with fewest wildcards
        if len(matches) > 1:
            matching_group = matches.loc[[(matches.filter(regex='atom_group?')
                                           == 0).sum(axis='columns').idxmin()]]
        elif len(matches) == 1:
            matching_group = matches
        else:
            # If there are no matches in the file, then raise an error
            msg = 'Unable to find bond information for specified atoms: '
            for i, atom_type in enumerate(groups):
                msg += f'atom_group{i} - {atom_type} ({self.atom_type_name[atom_type]}), '
            raise ValueError(msg[:-2])

        # Get the parameter names for the InteractionFunction. This means that
        # the columns in the bonded DataFrame do not need to be ordered,
        # as long as the column names are the same as the InteractionFunction
        # parameter names (e.g. equilibrium_state and potential_strength must
        # exist in self.bonds if it is a HarmonicPotential, but can be in any
        # order).
        parameter_names = self._get_parameter_names(function_type,
                                                    [name for name
                                                     in matching_group if
                                                     'atom_group' not in name])

        settings = {}
        if function_type == interaction_functions.HarmonicPotential:
            settings['interaction_type'] = bonded_type
        # As parameter_names is correctly sorted
        parameters = [getattr(matching_group, name).values[0] for name
                      in parameter_names]
        self._set_interaction_function(bonded, function_type, parameters,
                                       settings)

    def _parametrize_coulombic(self, coulombic):
        """
        Assumes that all force fields have a ``Coulomb`` ``InteractionFunction``

        Parameters
        ----------
        coulombic : Coulombic
            The ``Coulombic`` to be parametrized
        """

        # Check we have atoms to apply interaction to
        if len(coulombic.atoms) == 0:
            raise ValueError(f'Unable to find any atoms of types {coulombic.atom_types} '
                             'in the Universe to apply the Coulombic interaction to')

        # Different atom names could be defined within the same coulombic
        # Both atom name and element are required to uniquely identify the atom
        atom_names_elements = set()
        for atom in coulombic.atoms:
            self._convert_atom_type_name(atom)
            atom_names_elements.add((atom.name, atom.element.symbol))

        cols = ['name', 'element']
        matching_atoms = pd.concat([filter_ordered_dataframe(name_element,
                                                             self.atoms,
                                                             column_names=cols)
                                    for name_element in atom_names_elements])
        # Check that charges for all atom names are the same
        unique_charges = matching_atoms.charge.unique()
        self._check_nonbonded_parameters(unique_charges, ['charge'],
                                         matching_atoms)
        if len(unique_charges) != 1:
            # If not, show the corresponding atom rows in the error message
            msg = ('All atoms of the Coulombic interaction must have the same'
                   f' OPLS charge ({matching_atoms})')
            LOGGER.error('%s %s',
                         self.__class__,
                         msg)
            raise ValueError(msg)
        self._set_interaction_function(coulombic,
                                       interaction_functions.Coulomb,
                                       unique_charges)

    def _parametrize_dispersion(self, dispersion):
        """
        While dispersion interactions can be defined between unlike atom types,
        this is not the case for any major force field implementation (e.g.
        OPLS, CHARMM, AMBER). Therefore currently this only parametrizes
        like-like dispersion interactions.

        Parameters
        ----------
        dispersion : Dispersion
            The ``Dispersion`` to be parametrized
        """

        # As the atoms in a dispersion are determined by dispersion.atom_type,
        # each atom pair must be consistent (i.e. it is only necessary to
        # consider two atoms from each pair as they will be the same as all
        # other atoms in the pair)

        matching_disps = []
        for atom_pairs in dispersion.atoms:
            # Checks that atom_pairs are like-like and that we have atoms to
            # apply interaction to
            atom_pairs = list(atom_pairs)
            atom_pair = []
            for i, atom_type_pair in enumerate(atom_pairs):
                if len(atom_type_pair) == 0:
                    existing_types = []
                    for (key, value) in dispersion.universe.atom_types.items():
                        if len(value) > 0:
                            existing_types.append(key)
                    raise ValueError(f'No atoms of type "{dispersion.atom_types[0][i]}" '
                                     'found, the Universe contains only the types '
                                     f'{existing_types}')

                atom_pair.append(atom_type_pair[0])

            if len(set(atom_pair)) != 1:
                msg = ('Currently only force fields which only use like-like'
                       ' Dispersion terms are implemented')
                LOGGER.error('%s: {atom_pair: %s}. %s',
                             self.__class__,
                             atom_pair,
                             msg)
                raise ValueError(msg)

            # As only like-like, just consider first index of atom_pairs
            for atom in atom_pairs[0]:
                self._convert_atom_type_name(atom)
            # As all atoms within atom_pairs should be the same, only consider
            # single atom when determining ff_atom_type
            # ff_atom_type is different to MDMC atom_type
            atom_name_element = (atom_pairs[1][0].name,
                                 atom_pairs[1][0].element.symbol)
            cols = ['name', 'element']
            ff_atom_type = filter_ordered_dataframe(atom_name_element,
                                                    self.atoms,
                                                    column_names=cols).atom_type
            matching_disps.append(filter_dataframe(list(ff_atom_type),
                                                   self.dispersions,
                                                   column_names=['atom_type']))
        matching_disps = pd.concat(matching_disps)

        function_type = self._get_interaction_function('dispersion')
        parameter_names = self._get_parameter_names(function_type)

        unique_parameters = list(chain.from_iterable([matching_disps[n].unique()
                                                      for n in parameter_names])
                                 )
        self._check_nonbonded_parameters(unique_parameters, parameter_names,
                                         matching_disps)
        self._set_interaction_function(dispersion, function_type,
                                       unique_parameters)

    def _get_interaction_function(self, interaction_type):
        """
        Gets the ``InteractionFunction`` for the corresponding ``Interaction``

        Parameters
        ----------
        interaction_type : str
            A `str` specifying an interaction type, which must be a key in
            ``self._inter_functions``

        Returns
        -------
        InteractionFunction
            The subclass of ``InteractionFunction`` correspoding to
            ``interaction_type``
        """

        function_name = self.inter_functions[interaction_type.lower()]
        return getattr(interaction_functions, function_name)

    @weakref_cache(maxsize=None)
    def _convert_atom_type_name(self, atom):
        """
        Converts all ``Atom`` objects with ``Atom.name`` that are a valid force
        field type (i.e. can be cast to an `int`) into the corresponding force
        field atom name.

        Each ``Atom.name`` only needs to be converted once, however they will be
        converted for every interaction in which they appear - avoid this with
        lrucache

        Parameters
        ----------
        atom : Atom
            The ``Atom`` for which the name will be converted

        Raises
        ------
        ValueError
            If the ``Atom`` does not have a valid force field type or a force
            field atom name, raise a ValueError. This atom cannot be interpreted
            as valid within the force field.
        """

        # int cast as atom names will be strings, even if they are numerical
        # (e.g. an atom type)
        try:
            atom.name = self.atom_type_name[int(atom.name)]
        except (KeyError, ValueError) as err:
            # Check if atom.name is already a name in the atoms DataFrame
            if filter_dataframe([atom.name], self.atoms,
                                column_names=['name']).empty:
                msg = ('All atom names must be an OPLS atom type or an OPLS'
                       f' atom name. {atom.name} is not an OPLS atom type of atom'
                       ' name.')
                LOGGER.error('%s %s',
                             self.__class__,
                             msg,
                             exc_info=1)
                raise ValueError(msg) from err

    @staticmethod
    def _get_parameter_names(function, file_parameter_names=None):
        """
        Gets the names of the parameters of function, excluding ``self`` and
        ``**settings``

        If ``*args`` is included in the function signature, then the parameter
        names are taken from the file.

        Parameters
        ----------
        function : callable
            Any callable (e.g. function, method)
        file_parameter_names : list
            `list` of `str` with the parameter names from the file

        Returns
        -------
        `list` of `str`
            The parameters of the function

        Raises
        ------
        ValueError
            If the function signature has ``*args`` and the parameter names in
            the file are incorrectly ordered.
        """

        # VAR_POSITIONAL is the Python Parameter kind which is used for *args
        var_positional = False
        parameter_names = []
        for function_parameters in signature(function).parameters.values():
            # Test that the kind of parameter is a VAR_POSITIONAL
            if function_parameters.kind == function_parameters.VAR_POSITIONAL:
                var_positional = True
            else:
                name = function_parameters.name
                if name not in ['self', 'settings']:
                    parameter_names.append(name)

        if var_positional:
            # If there is *args in the function signature, we cannot determine
            # the parameter names. Check that the parameter names in the
            # signature agree with the order of the parameter names in the file.
            # If this is the case, assume file parameter names are correctly
            # ordered and use these, otherwise raise a ValueError
            if any(parameter_names[i] != file_parameter_names[i] for i
                    in range(len(parameter_names))):
                msg = (f'The force field data file has incorrectly ordered {parameter_names}'
                       ' parameters')
                LOGGER.error('FileForceField: %s',
                             msg)
                raise ValueError(msg)
            parameter_names = file_parameter_names

        return parameter_names

    @staticmethod
    def _check_nonbonded_parameters(function_parameters,
                                    function_parameter_names,
                                    matching_inters):
        """
        Checks that the parameters for an ``InteractionFunction`` for a
        ``NonBondedInteraction`` are valid

        Parameters
        ----------
        function_parameters : list
            The parameters which will be passed to an ``InteractionFunction``
        function_parameter_names : list
            A list of the names of the ``function_parameters``
        matching_inters : pandas.DataFrame
            A ``Dataframe`` of the lines of the interaction in the force field
            file which match the ``NonBondedInteraction`` which is being
            parametrized

        Raises
        ------
        ValueError
            If atoms which have been added to the ``NonBondedInteraction`` do
            not have the same ``InteractionFunction`` parameters for the force
            field, they cannot be included in the same ``NonBondedInteraction``.
        """

        if len(function_parameters) != len(function_parameter_names):
            msg = ('All atoms of the interaction must have the same OPLS'
                   f' parameters ({matching_inters})')
            LOGGER.error('FileForceField: %s',
                         msg)
            raise ValueError(msg)

    @staticmethod
    def _set_interaction_function(interaction, function_type,
                                  function_parameters, function_settings=None):
        """
        Initialises an ``InteractionFunction`` with specified parameters and
        settings and sets it for an ``Interaction``

        Parameters
        ----------
        interaction : Interaction
            The ``Interaction`` for which the ``InteractionFunction`` will be
            set
        function_type : InteractionFunction class
            The ``InteractionFunction`` class which will be initialised
        function_parameters : list
            The parameters which are passed to the ``function_type``
        function_settings : dict, optional
            Any ``**settings`` to be passed to the ``function_type`` for
            initialising. The default is `None`, which results in no
            ``**settings`` being passed.
        """

        function_settings = function_settings if function_settings else {}
        interaction.function = function_type(*function_parameters,
                                             **function_settings)
        interaction.function.set_parameters_interactions(interaction)

    @staticmethod
    def _parse_header(header, dtype):
        """
        Parses a force field file header line

        Parameters
        ----------
        header : str
            The header line to be parsed
        dtype : class
            The expected datatype of the ``header`` parameters, e.g. `int` or
            `str`

        Returns
        -------
        list of tuples
            (``keyword``, ``parameter``) where ``keyword`` is a `str` which
            specifies a force field file section, and ``parameter`` is the
            description associated with it.
        """

        # Strip newline formatting
        return [(name.strip('\n').lower(), dtype(parameter.strip('\n')))
                for name, parameter in [s.split('=') for s in header.split(' ')]]

    @staticmethod
    def _set_col_type(column):
        """
        Sets the type of a ``DataFrame`` column to either a `float` or an `int`,
        if possible

        Parameters
        ----------
        column : pandas.Series
            The column for which the type will be converted
        """

        if any(column.astype(str).str.contains(escape('.'))):
            try:
                return column.astype(float)
            except ValueError:
                return column
        else:
            try:
                return column.astype(int)
            except ValueError:
                return column

"""Module for calculating the pair distribution function (PDF)"""

from collections import defaultdict
from itertools import (chain, combinations, combinations_with_replacement,
                       product)
import warnings

from mpi4py import MPI, futures
from numba import jit
import numpy as np

from MDMC.common.atom_properties import B_COH
from MDMC.trajectory_analysis.observables.obs import Observable


class PairDistributionFunction(Observable):

    """
    A class for containing, calculating and reading a pair distribution
    function
    """

    @property
    def data(self):

        return {'independent':self.independent_variables,
                'dependent':self.dependent_variables,
                'errors':self.errors}

    @property
    def independent_variables(self):

        """
        Get or set the independent variables, the atomic separation distance r
        (in Ang)

        Returns
        -------
        dict
            The independent_variables
        """

        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, value):

        self._independent_variables = value

    @property
    def dependent_variables(self):

        """
        Get or set the dependent variables, the pair distribution function (in
        arb)

        Returns
        -------
        dict
            The dependent_variables
        """

        return self._dependent_variables

    @property
    def errors(self):

        """
        Get or set the errors on the dependent variables, the pair distribution
        function (in arb)

        Returns
        -------
        dict
            The errors on the dependent variables
        """

        return self._errors

    @errors.setter
    def errors(self, value):

        self._errors = value

    @property
    def r(self):

        """
        Get the value of the atomc separation distance (in Ang)
        """

        try:
            return self.independent_variables['r']
        except KeyError:
            raise AttributeError

    def calculate_from_MD(self, MD_input, **settings):

        """
        Calculate the pair distribution function, G(r) from a MD Trajectory

        Independent variables can either be set previously or defined within
        settings.

        A number of frames can be specified, from which the pdf and its error
        are calculated. If the number of frames is too large relative to the run
        length, the samples will be correlated, which will result in an
        underestimate of the error.

        Parameters
        ----------
        MD_input : Trajectory
            An MD Trajectory
        **settings
            n_frames : int
                The number of frames from which the pdf and its error are
                calculated. If this is not passed, 1% of the total number of
                frames are used (rounded up to nearest int).
            subset : list of tuples
                The subset of element pairs from which the PDF is calculated.
                This can be used to calculate the partial PDFs of a
                multicomponent system. If this is not passed, all combinations
                of elements are used i.e. the PDF is the total PDF.
            b_coh : dict
                Definitions of the coherent neutron scattering lengths for one
                or more elements. This can be used to calculate the PDF of a
                system where one or more elements has a coherent scattering
                length different from the coherent scattering length in
                MDMC.common.atom_properties (i.e. if it has been isotopically
                substituted).
            r_min : float
                The minimum r (atomic separation) for which the PDF will be
                calculated. If this, r_max, and r_step are passed then these
                will create a range for the independent variable r, which will
                overwrite any r which has previously been defined.
            r_max : float
                The maximum r (atomic separation) for which the PDF will be
                calculated. If this, r_min, and r_step are passed then these
                will create a range for the independent variable r, which will
                overwrite any r which has previously been defined.
            r_step : float
                The step size of r (atomic separation) for which the PDF will be
                calculated. If this, r_min, and r_max are passed then these
                will create a range for the independent variable r, which will
                overwrite any r which has previously been defined.

        Examples
        --------
        To calculate the O-O partial PDF from a simulation of water, use the
        subset keyword:

            .. highlight:: python
            .. code-block:: python

            pdf.calculate_from_MD(trajectory, subset=[(O, O)])

        To calculate the sum of the H-O and O-O partial PDFs:

            .. highlight:: python
            .. code-block:: python

            pdf.calculate_from_MD(trajectory, subset=[(O, O), (H, O)])

        To calculate the total PDF for sodium chloride with 37Cl:

            .. highlight:: python
            .. code-block:: python

            pdf.calculate_from_MD(trajectory, b_coh={'Cl':3.08}

        To calculate the total PDF for r values of [1., 2., 3., 4., ]:

            .. highlight:: python
            .. code-block:: python

            pdf.calculate_from_MD(trajectory, b_coh={'Cl':3.08}
        """

        self.origin = 'MD'
        self._parse_calc_MD_settings(MD_input, settings)

        for trajectory in self.trajectory:
            # Filter the atoms so that we are only considering elements included
            # in self.elements


            # Partition the atoms in each frame of the reduced trajectory so
            # that only atoms within r_max of each other are paired (as atom
            # pairs with separations > r_max cannot contribute to pdf)
            # trajectory is of a single frame so will only have 1
            # configuration (i.e. index zero)
            self._calculate_histogram(trajectory.configurations[0])

        # Sum the partials and scale by the remainder of the prefactor (e.g.
        # anything not element dependent)
        prefactor = 1 / (4.0 * np.pi * self.r**2 * self.r_step)
        for elements, partial in self.partial_pdfs.items():
            weighting = np.multiply(*[self.weights[elem] for elem in elements])
            partial *= (prefactor * weighting)

    def _parse_calc_MD_settings(self, trajectory, settings):

        """
        Parses the MD settings

        This includes setting:
            - the number of evenly spaced frames for which the PDF will be
              averaged
            - the partial pairs for which PDF will be calculated
            - the elements involved in the PDF
            - the weights for each element
            - the volume of the universe
            - the independent variables (r)

        Parameters
        ----------
        settings : dict
            A dictionary of settings to be parsed

        Warnings
        --------
        UserWarning
            If one or two of r_min, r_max, and r_step have been passed, user is
            warned that three are required to set r.
        """

        # np.max ensures that n_frames is at least 1 (relevant if
        # total_n_frames < 100)
        total_n_frames = len(trajectory)
        n_frames = settings.get('n_frames', np.max([1, total_n_frames // 100]))
        # If only a single frame then set frame_step > total_n_frames
        frame_step = (total_n_frames + 1 if n_frames == 1
                      else total_n_frames // (n_frames - 1))
        self.trajectory = trajectory[0:total_n_frames:frame_step]

        self.partial_strings = settings.get('subset', \
            list(combinations_with_replacement(trajectory.element_set, 2)))

        # Create element set from elements in partials. The weights are then
        # determined from these.
        self.elements = set(chain.from_iterable(self.partial_strings))
        self.weights = self._set_weights(self.elements,
                                         settings.get('b_coh', {}))
        self.numbers = self._set_numbers(self.elements,
                                         self.trajectory.element_list)

        self.universe_dimensions = np.array((settings.get('dimensions')
                                             or trajectory.dimensions))
        self.universe_volume = np.prod(self.universe_dimensions)
        # PDF only valid where number of atoms is conserved over trajectories
        self.n_atoms = len(self.trajectory[0].atoms)

        # Create independent_variables dictionary if it doesn't exist
        if not hasattr(self, 'independent_variables'):
            self.independent_variables = {}

        # If rmin, rmax and rstep are in settings, overwrite existing values for
        # independent variable. If one or two are in settings, warn the user
        # that all three are required to set r.
        r_kwargs = ['r_min', 'r_max', 'r_step']
        if all(r_kw in settings.keys() for r_kw in r_kwargs):
            self.independent_variables['r'] = np.arange(settings['r_min'],
                                                        settings['r_max'] +
                                                        settings['r_step'],
                                                        settings['r_step'])
        elif any(r_kw in settings.keys() for r_kw in r_kwargs):
            warnings.warn('Setting r requires r_min, r_max and r_step to all be'
                          ' set. Using existing r instead.')
        self.r_step = self.r[1] - self.r[0]

        self.partial_pdfs = {partial_string:
                             np.zeros(np.shape(self.independent_variables['r']))
                             for partial_string in self.partial_strings}

        # Release memory from full trajectory
        del trajectory

    def _calculate_histogram(self, configuration):

        """
        Partitions the atomic positions into regions where they are within r_max
        from all other atoms
        """

        def get_component_lengths(universe_dim):

            """
            Use r values for each component that are at least as big as r_max,
            but that are a factor of the dimensions

            When r > r_max, some atom pairs within each partition will have
            separations which will not be histogrammed (as they are > r_max),
            however this is unavoidable
            """

            r_max = np.max(self.independent_variables['r'])
            return universe_dim / (universe_dim // r_max)

        part_comps = np.array(list(map(get_component_lengths,
                                       self.universe_dimensions)))
        partitions = self._partition(configuration.atom_positions,
                                     configuration.element_list,
                                     part_comps)
        # Get the partition_indexes and the pairs of partitions. As well as
        # calculating atom pairs within a partition, each partition will have 26
        # neighbors for which atom pairs must be calculated.
        partition_indexes = list(self._calculate_partition_indexes(part_comps))
        partition_pair_indexes = self._get_partition_pairs(part_comps)

        # Calculate the histograms of all atoms in each partition for element
        # combinations that are in self.partial_strings
        for partial_string in self.partial_strings:
            elem1, elem2 = partial_string
            like_elems = elem1 == elem2
            pos_pairs = []

            # Get the atom pairs for atoms in the same partition
            for part_i in partition_indexes:
                if like_elems:
                    # combinations avoids an atom and itself being an atom pair
                    pos_pairs.append(combinations(partitions[elem1][part_i], 2))
                else:
                    # atom and itself as an atom pair not an issue for unlike
                    # elements
                    pos_pairs.append(product(partitions[elem1][part_i],
                                             partitions[elem2][part_i]))

            # Get the atom pairs for atoms in different partitions
            for part1, part2 in partition_pair_indexes:
                # Correct for periodic boundary conditions, which will only
                # occur if modulus of separation distance in any direction is
                # greater than 1 (i.e. the partitions would not be neighbors
                # without pdb)
                wrap = np.zeros([3])
                for i in range(3):
                    component_separation = part1[i] - part2[i]
                    if component_separation > 1:
                        wrap[i] = 1
                    elif component_separation < -1:
                        wrap[i] = -1
                wrap *= self.universe_dimensions

                # try/excepts are in case partitions[elem1][part1] is empty
                try:
                    pos_pairs.append(product(partitions[elem1][part1] - wrap,
                                             partitions[elem2][part2]))
                except ValueError:
                    pass
                # For unlike elements, also consider other element/partition
                # combination
                if not like_elems:
                    try:
                        pos_pairs.append(product(partitions[elem2][part1] -wrap,
                                                 partitions[elem1][part2]))
                    except ValueError:
                        pass
            # pos_pairs is a list of iterators - flatten this to a single
            # iterator
            pos_pairs = chain(*pos_pairs)
            self.partial_pdfs[partial_string] += \
                self._calculate_histogram_from_position_pairs(pos_pairs)

    def _partition(self, positions, element_list, part_comps):
        # Set up a partitions dictionarty separated by element
        partitions = {element:defaultdict(list) for element
                      in self.elements}

        # Add empty lists for all possible partition indexes. This will allow
        # product and combinations to include these indexes (although they
        # will be empty combinations and products)
        for element in self.elements:
            for part_index in self._calculate_partition_indexes(part_comps):
                partitions[element][part_index] = []

        # Drop positions of atoms of elements not included
        mask = np.isin(element_list, list(self.elements))
        element_array = np.array(element_list)[mask]
        positions = positions[mask]

        # Get element and position of each atom
        for elem, position in zip(element_array, positions):
            partition_index = []
            for component, part_comp in zip(position, part_comps):
                partition_index.append(component // part_comp)
            # Add each position to correct partition
            partitions[elem][tuple(partition_index)].append(position)
        # Convert defaultdicts(list) to dicts of numpy arrays (just changes type
        # of positions from list to array)
        return {elem:{partition_index:np.array(positions)
                      for partition_index, positions in elem_partitions.items()}
                for elem, elem_partitions in partitions.items()}

    def _get_partition_pairs(self, partition_components):

        p_indexes = self._calculate_partition_indexes(partition_components)
        max_indices = (self.universe_dimensions
                       / partition_components).astype('int32') - 1
        identities = ([(-1, 1, 0), (-1, 1, 1), (-1, 1, -1),
                       (1, 0, -1), (0, 1, -1), (1, 1, -1)]
                      + list(product(*map(np.arange, (2, 2, 2))))[1:])
        pairs = []
        for p_index in p_indexes:
            select_neighbors = [np.add(p_index, iden) for iden in identities]
            for neighbor in select_neighbors:
                above_upper_bound = np.where(neighbor > max_indices)
                neighbor[above_upper_bound] = 0
                below_lower_bound = np.where(neighbor < 0)
                neighbor[below_lower_bound] = max_indices[below_lower_bound]
                pairs.append((p_index, tuple(neighbor)))

        return pairs

    def _calculate_partition_indexes(self, partition_components):

        return product(*map(np.arange, (self.universe_dimensions
                                        / partition_components).astype('int32'))
                      )

    def _calculate_histogram_from_position_pairs(self, position_pairs):

        """
        Returns a histogram of pair separations calculated from position_pairs

        Parameters
        ----------
        position_pairs : nd.array
            An array of 2 element tuples, where each element is a 3 element
            vector specifying a position.

        Returns
        -------
        nd.array
            An histogram of length equal to the length of r in
            independent_variables, where each count is an atomic separation
            (between the elements in a tuple in position_pairs)
        """

        # Use np.histogram to get empty array of correct size and bin edges
        # Assumes constant r step size
        r_step = self.r[1] - self.r[0]
        r_min, r_max = np.min(self.r) - r_step / 2, np.max(self.r) + r_step / 2
        hist, bin_edges = np.histogram([], len(self.r), range=(r_min, r_max))

        @jit('float64[:], float64[:]', nopython=True)
        def jit_histogram(separations, bin_edges):
            jit_hist, _ = np.histogram(separations, bins=bin_edges)
            return jit_hist

        # Calculate histograms over blocks. This is both for memory management
        # and to enable scattering for MPI. Numpy also uses blocks, with a BLOCK
        # size of 65536 (256**2), so will use this block size and pad with zeros
        # if position_pairs has been exhausted.
        # pylint: disable=invalid-name
        BLOCK = 65536
        exhausted = False
        while not exhausted:
            # Using next with default means pair_block will be filled with
            # np.array(['inf']) if StopIteration is returned. Then change
            # exhausted so that the while loop will stop.
            pair_block = [np.subtract(*next(position_pairs, ([np.float('inf')],
                                                             [0.])))
                          for _ in range(BLOCK)]
            exhausted = np.float('inf') in pair_block[-1]
            separations = map(self._calculate_separation, pair_block)
            hist += jit_histogram(np.fromiter(separations,
                                              dtype=np.float64,
                                              count=BLOCK),
                                  bin_edges)
        return hist

        # Calculate the seperation distances
        # Should be possible to use numba for this
        # Use mpi4py to parallelise - split up the total number of atoms
        # After each separation distance is calculated, bin it in the
        # histogram. This is done in preference to using np.histogram,
        # as otherwise all atom separations will need to be in memory
        # simultaneously - for large systems this will be prohibitive,
        # as it will scale with N!

        # Calculate the element dependent prefactor and scale by this
        # Use settings.get to check if settings has b_cohs defined

    @staticmethod
    @jit('float64(float64[:])', nopython=True)
    def _calculate_separation(positions):

        """
        numba.jit results in ~10x speed up over np.linalg.norm
        """

        return np.sum(positions ** 2) ** 0.5

    @staticmethod
    def _set_weights(unique_elements, b_coh):

        """
        Sets the weights for each element

        Uses any scattering lengths passed in b_coh, but defaults to values in
        MDMC.common.atom_properties

        Parameters
        ----------
        unique_elements : list of str
            Where each str specifies an element
        b_coh : dict
            (element:b_c) where element is a str specifying an element occuring
            in unique_elements, and b_c is the weight (coherent scattering
            length) to be set for that element

        Returns
        -------
        dict
            (element:weight) where element is a str and weight is the
            corresponding weight
        """

        return {element:b_coh.get(element, B_COH[element]) for element
                in unique_elements}

    @staticmethod
    def _set_numebrs(unique_elements, element_list):

        """
        Sets the number of atoms of each element

        Parameters
        ----------
        unique_elements : list of str
            Where each str specifies an element
        element_list : list of str
            A list of the elements for every atom in the trajectory

        Returns
        -------
        dict
            (element:number) where element is a str and number is the number of
            atoms of the that element in the element list
        """

        return {element:element_list.count(element) for element
                in unique_elements}

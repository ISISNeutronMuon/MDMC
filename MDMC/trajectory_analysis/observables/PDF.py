"""Module for calculating the pair distribution function (PDF)"""

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
        self.trajectory = MD_input
        self._parse_calc_MD_settings(settings)

        for partial_string in self.partial_strings:
            for trajectory in self.reduced_trajectory:
                # trajectory is of a single frame so will only have 1
                # configuration (i.e. index zero)
                atom_pairs = self._get_atom_pairs(trajectory.configurations[0],
                                                  partial_string)
                self.partial_pdfs[partial_string] += \
                    self._calculate_histogram(atom_pairs)


        # Sum the partials and scale by the remainder of the prefactor (e.g.
        # anything not element dependent)

    def _parse_calc_MD_settings(self, settings):

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
        total_n_frames = len(self.trajectory)
        n_frames = settings.get('n_frames', np.max([1, total_n_frames // 100]))
        # If only a single frame then set frame_step > total_n_frames
        frame_step = (total_n_frames + 1 if n_frames == 1
                      else total_n_frames // (n_frames - 1))
        self.reduced_trajectory = self.trajectory[0:total_n_frames:frame_step]

        self.partial_strings = settings.get('subset', \
            list(combinations_with_replacement(self.trajectory.element_set, 2)))

        # Create element set from elements in partials. The weights are then
        # determined from these.
        self.elements = set(chain.from_iterable(self.partial_strings))
        self.weights = self._set_weights(self.elements,
                                         settings.get('b_coh', {}))

        dimensions = settings.get('dimensions') or self.trajectory.dimensions
        self.universe_volume = np.prod(dimensions)

        # Create independent_variables dictionary if it doesn't exist
        if not hasattr(self, 'independent_variables'):
            self.independent_variables = {}

        # If rmin, rmax and rstep are in settings, overwrite existing values for
        # independent variable. If one or two are in settings, warn the user
        # that all three are required to set r.
        r_kwargs = ['r_min', 'r_max', 'r_step']
        if all(r_kw in settings.keys() for r_kw in r_kwargs):
            self.independent_variables['r'] = np.arange(settings['r_min'],
                                                        settings['r_max'],
                                                        settings['r_step'])
        elif any(r_kw in settings.keys() for r_kw in r_kwargs):
            warnings.warn('Setting r requires r_min, r_max and r_step to all be'
                          ' set. Using existing r instead.')

        self.partial_pdfs = {partial_string:
                             np.zeros(np.shape(self.independent_variables['r']))
                             for partial_string in self.partial_strings}

    def _get_atom_pairs(self, configuration, partial):

        """
        Gets the positions of the atom pairs for which a histogram of pair
        separations can be calculated

        Parameters
        ----------
        configuration : Configuration
            The configuration for which the atom pair positions will be
            determined
        partial : tuple
            (element1, element2) where element1 and element2 are strings
            specifying the elements of the partial pair for which the atom pair
            positions will be determined

        Returns
        -------
        iterator
            An iterator of tuples where each tuple contains a pair of atom
            positions for which the separation should be calculated in order to
            determine the histogram for the partial pair
        """

        el1_positions = self._get_elem_positions(configuration.atom_positions,
                                                 configuration.element_list,
                                                 partial[0])

        # For partials with a single element duplciated, the atom pairs are
        # every atom with every other atom
        if partial[0] == partial[1]:
            return combinations(el1_positions, 2)
        # For partials with different elements, the atom pairs are every atom of
        # one element with every atom of the other element
        el2_positions = self._get_elem_positions(configuration.atom_positions,
                                                 configuration.element_list,
                                                 partial[1])
        return product(el1_positions, el2_positions)

    def _calculate_histogram(self, atom_pairs):

        """
        Returns a histogram of pair separations calculated from atom_pairs

        Parameters
        ----------
        atom_pairs : nd.array
            An array of 2 element tuples, where each element is a 3 element
            vector specifying a position.

        Returns
        -------
        nd.array
            An histogram of length equal to the length of r in
            independent_variables, where each count is an atomic separation
            (between the elements in a tuple in atom_pairs)
        """

        # Use np.histogram to get empty array of correct size and bin edges
        r_min = np.min(self.independent_variables['r'])
        r_max = np.max(self.independent_variables['r'])
        hist, bin_edges = np.histogram([],
                                       len(self.independent_variables['r']),
                                       range=(r_min, r_max))

        @jit('float64[:], float64[:]', nopython=True)
        def jit_histogram(separations, bin_edges):
            jit_hist, _ = np.histogram(separations, bins=bin_edges)
            return jit_hist

        # Calculate histograms over blocks. This is both for memory management
        # and to enable scattering for MPI. Numpy also uses blocks, with a BLOCK
        # size of 65536 (256**2), so will use this block size and pad with zeros
        # if atom_pairs has been exhausted. As long as there are at least 363
        # atoms in the calculation, there will be sufficient atom pairs to fill
        # at least one block (for the same-same partial).
        # pylint: disable=invalid-name
        BLOCK = 65536
        exhausted = False
        while not exhausted:
            # Using next with default means pair_block will be filled with
            # (0., 0.) if StopIteration is returned. Then change exhauted so
            # that the while loop will stop.
            pair_block = [np.subtract(*next(atom_pairs, (np.array([0.]),
                                                         np.array([0.]))))
                          for _ in range(BLOCK)]
            print(pair_block[-1])
            exhausted = all(pair_block[-1] == np.array([0.]))
            with futures.MPIPoolExecutor(MPI.COMM_WORLD.size) as executor:
                separations = executor.map(self._calculate_separation, pair_block)
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
    def _get_elem_positions(config_positions, config_element_list, element):

        """
        Get positions from a configuration for a single element

        Parameters
        ----------
        config_positions : nd.array
            The positions of all of the atoms in the configuration, with the
            equivalent order to config_element_list
        config_element_list : nd.array
            The elements of each of the atoms which have the positions specicied
            in config_positions
        element : str
            The element for which the positions will be returned

        Returns
        -------
        nd.array
            Array of positions for atoms of element
        """

        indexes = np.where(np.array(config_element_list) == element)
        return np.array(config_positions[indexes])

    @staticmethod
    def _set_weights(elements, b_coh):

        """
        Sets the weights for each element

        Uses any scattering lengths passed in b_coh, but defaults to values in
        MDMC.common.atom_properties

        Parameters
        ----------
        elements : list of str
            Where each str specifies an element
        b_coh : dict
            (element:b_c) where element is a str specifying an element occuring
            in elements, and b_c is the weight (coherent scattering length) to
            be set for that element

        Returns
        -------
        dict
            (element:weight) where element is a str and weight is the
            corresponding weight
        """

        return {element:b_coh.get(element, B_COH[element]) for element
                in elements}

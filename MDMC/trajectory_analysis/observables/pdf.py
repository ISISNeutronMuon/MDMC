"""Module for calculating the total pair distribution function (PDF)"""
from collections import defaultdict
from itertools import (chain, combinations_with_replacement, islice,
                       product)
from typing import Optional
import warnings

import numpy as np

from MDMC.common.atom_properties import B_COH
from MDMC.common import units
from MDMC.common.decorators import unit_decorator, unit_decorator_getter
from MDMC.trajectory_analysis.observables.obs import Observable
from MDMC.trajectory_analysis.observables.obs_factory import ObservableFactory
from MDMC.trajectory_analysis.compact_trajectory import CompactTrajectory


@ObservableFactory.register(('PDF', 'PairDistributionFunction'))
class PairDistributionFunction(Observable):
    r"""
    A class for containing, calculating and reading a pair distribution function (PDF).

    We derive our definitions for this from the following publication:
    "A comparison of various commonly used correlation functions for describing total scattering"
    Keen, D. A. (2001). J. Appl. Cryst. 34, 172-177.
    DOI: https://doi.org/10.1107/S0021889800019993

    We employ the following mathematical form for the total pair distribution function (``PDF``):

        .. math::

            G(r) = \sum_{i,j}^{N_{elements}} c_ic_jb_ib_j(g_{ij}(r) - 1)


        where :math:`c_i` is the number concentration of element :math:`i`,
        :math:`b_i` is the (coherent) scattering length of element :math:`i`.
        (This corresponds to equation 8 in the above publication)


    The partial pair distribution, :math:`g_{ij}`, is:

        .. math::

            g_{ij}(r) = \frac{h_{ij}(r)}{4 \pi r^2 \rho_{j} \Delta{r}}

        where :math:`h_{ij}`` is the histogram of distances of :math:`j` element
        atoms around atoms of element :math:`i`, with bins of size
        :math:`\Delta{r}`, and :math:`\rho_{j}` is the number density of
        atoms of element :math:`j`. As :math:`g_{ij}(0) = 0`, it is evident that
        :math:`G(0) = -\sum_{i,j}^{N_{elements}} c_ic_jb_ib_j`.
        (This corresponds to equation 10 in the above publication)

    The total PDF is contained in ``PDF`` and the partial pair PDFs (if calculated or imported)
    are contained in ``partial_pdfs``.
    """

    def __init__(self):

        super().__init__()
        self._independent_variables = None
        self._dependent_variables = None
        self._errors = None
        self.partial_strings = None
        self.elements = None
        self.weights = None
        self.numbers_of_atoms = None
        self.universe_volume = None
        self.n_atoms = None
        self.r_step = None
        self.partial_pdfs = None

    @property
    def independent_variables(self) -> dict:
        """
        Get or set the independent variable: this is
        the atomic separation distance r (in ``Ang``)

        Returns
        -------
        dict
            The independent variables
        """

        return self._independent_variables

    @independent_variables.setter
    def independent_variables(self, value: dict) -> None:
        self._independent_variables = value

    @property
    def dependent_variables(self) -> dict:
        """
        Get the dependent variables: these are PDF, the pair distribution function (in ``barn``)

        Returns
        -------
        dict
            The dependent variables
        """

        return self._dependent_variables


    @property
    def errors(self) -> dict:
        """
        Get or set the errors on the dependent variables, the pair distribution function
        (in ``arb``)

        Returns
        -------
        dict
            The errors on the ``dependent_variables``
        """

        return self._errors

    @errors.setter
    def errors(self, value: dict) -> None:
        self._errors = value

    def minimum_frames(self, dt: float = None) -> int:
        """
        The minimum number of ``CompactTrajectory`` frames needed to
        calculate the ``dependent_variables`` is 1

        Parameters
        ----------
        dt : float, optional
            The time separation of frames in ``fs``, default is `None`, not
            used

        Returns
        -------
        int
            The minimum number of frames
        """

        return 1

    def maximum_frames(self) -> None:
        """
        There is no hard limit on the number of frames that can be used, so
        return None

        Returns
        -------
        None
        """

        return None

    @property
    def r(self) -> Optional[float]:
        """Get or set the value of the atomic separation distance (in ``Ang``)"""
        try:
            return self.independent_variables['r']
        except KeyError:
            return None

    @r.setter
    @unit_decorator(unit=units.Unit('Ang'))
    def r(self, value: float) -> None:
        if (hasattr(self, '_independent_variables')
                and self._independent_variables):
            self._independent_variables['r'] = value
        else:
            self._independent_variables = {'r': value}

    @property
    @unit_decorator_getter(unit=units.Unit('barn'))
    def PDF(self) -> Optional[float]:
        """Get the value of the total pair distribution function (in ``barn``)"""
        try:
            return self._dependent_variables['PDF']
        except KeyError:
            return None

    @property
    @unit_decorator_getter(unit=units.Unit('barn'))
    def PDF_err(self) -> Optional[float]:
        """Get the errors on the total pair distribution function (in ``barn``)"""
        try:
            return self.errors['PDF']
        except KeyError:
            return None


    def calculate_from_MD(self, MD_input: CompactTrajectory, verbose: int = 0, **settings: dict):
        r"""
        Calculate the pair distribution function, :math:`G(r)`` from a
        ``CompactTrajectory``

        The partial pair distribution for a pair i-j, :math:`g_{ij}`, is:

        .. math::

            g_{ij}(r) = \frac{h_{ij}(r)}{4 \pi r^2 \rho_{j} \Delta{r}}

        where :math:`h_{ij}`` is the histogram of distances of :math:`j` element
        atoms around atoms of element :math:`i`, with bins of size
        :math:`\Delta{r}`, and :math:`\rho_{j}` is the number density of
        atoms of element :math:`j`. As :math:`g_{ij}(0) = 0`, it is evident that
        :math:`G(0) = -\sum_{i,j}^{N_{elements}} c_ic_jb_ib_j`.

        This corresponds to the equation (8) in the following paper:
        "A comparison of various commonly used correlation functions for
         describing total scattering"
        Keen, D. A. (2001). J. Appl. Cryst. 34, 172-177.
        DOI: https://doi.org/10.1107/S0021889800019993

        The total pair distribution function (``pdf.PDF``) has the form:

        .. math::

            G(r) = \sum_{i,j}^{N_{elements}} c_ic_jb_ib_j(g_{ij}(r) - 1)

        where :math:`c_i` is the proportion of element :math:`i` in the material,
        :math:`b_i` is the (coherent) scattering length of element :math:`i`

        This corresponds to the equation (10) in the above paper.

        Independent variables can either be set previously or defined within
        settings.

        A number of frames can be specified, from which the ``PDF`` and its
        error are calculated. If the number of frames is too large relative to
        the run length, the samples will be correlated, which will result in an
        underestimate of the error.

        Parameters
        ----------
        MD_input : CompactTrajectory
             A single ``CompactTrajectory`` object.
        verbose: int
            Verbose print settings. Not currently implemented for PDF.
        **settings
            n_frames : int
                The number of frames from which the pdf and its error are
                calculated. These frames are selected uniformly, and the step is taken to be
                n_frames / total number of frames rounded to the nearest positive integer.
                If this is not passed, 1% of the total number of frames are used
                (rounded up to the nearest positive integer).
            use_average : bool
                Optional parameter. If set to True then the mean value for PDF is
                calculated across selected frames from the trajectory. Also, the errors
                are set to the standard deviation calculated over the multiple frames.
                If set to False (default), only the last frame of the trajectory is used and
                n_frames will be ignored.
            subset : list of tuples
                The subset of element pairs from which the PDF is calculated.
                This can be used to calculate the partial PDFs of a
                multicomponent system. If this is not passed, all combinations
                of elements are used i.e. the PDF is the total PDF.
            b_coh : dict
                A dictionary containing coherent scattering length values for one or more elements.
                This can be used to calculate the PDF of a system where one or more elements
                has a coherent scattering length different from the coherent scattering length in
                MDMC.common.atom_properties (i.e. if it has been isotopically substituted).
            r_min : float
                The minimum ``r`` (atomic separation) in Angstrom for which the PDF will be
                calculated. If this, ``r_max``, and ``r_step`` are passed then
                these will create a range for the independent variable ``r``,
                which will overwrite any ``r`` which has previously been
                defined. This cannot be passed if ``r`` is passed.
            r_max : float
                The maximum ``r`` (atomic separation) in Angstrom for which the PDF will be
                calculated. If this, ``r_min``, and ``r_step`` are passed then
                these will create a range for the independent variable ``r``,
                which will overwrite any ``r`` which has previously been
                defined. This cannot be passed if ``r`` is passed.
            r_step : float
                The step size of ``r`` (atomic separation) for which the PDF
                will be calculated. If this, ``r_min``, and ``r_max`` are passed
                then these will create a range for the independent variable
                ``r``, which will overwrite any ``r`` which has previously been
                defined. This cannot be passed if ``r`` is passed.
            r : numpy.ndarray
                The uniform ``r`` values in Angstrom for which the PDF will be calculated.
                This cannot be passed if ``r_min``, ``r_max``, and ``r_step``
                are passed.
            dimensions : array-like
                A 3 element `array-like` (`list`, `tuple`) with the dimensions
                of the ``Universe`` in Angstrom.


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

            pdf.calculate_from_MD(trajectory, b_coh={'Cl':3.08})

        To calculate the total PDF for r values of [1., 2., 3., 4.]:

            .. highlight:: python
            .. code-block:: python

            pdf.calculate_from_MD(trajectory, r=[1., 2., 3., 4.])

        To calculate the total PDF over an average of 5 frames:

            .. highlight:: python
            .. code-block:: python

            pdf.calculate_from_MD(trajectory, use_average=True, n_frames=5)
        """

        self.origin = 'MD'
        use_average = settings.get('use_average', False)
        self._parse_apply_MD_settings(MD_input, settings)

        if not use_average:
            self._calculate_partial_pdfs(self.trajectory)
            self._calculate_total_pdf()
            return

        running_partial_total = {}
        pdf_running_total = np.zeros(shape=len(self.r))

        n_frames = self.trajectory.n_steps
        # Calculate partial and total PDF for each frame in the sliced trajectory
        for frame in self.trajectory:
            self._calculate_partial_pdfs(frame)
            for partial_name in self.partial_pdfs:
                if partial_name in running_partial_total:
                    running_partial_total[partial_name] += self.partial_pdfs[partial_name]
                else:
                    running_partial_total[partial_name] = self.partial_pdfs[partial_name]
            self._calculate_total_pdf()
            pdf_running_total += self._dependent_variables['PDF']

        # Average over number of frames used in the sliced trajectory
        for partial_name in running_partial_total:
            self.partial_pdfs[partial_name] = np.divide(self.partial_pdfs[partial_name], n_frames)
        self._dependent_variables['PDF'] = np.divide(pdf_running_total, n_frames)

    def _calculate_partial_pdfs(self, trajectory: CompactTrajectory) -> None:
        """
        Calculate the partial PDFs for each partial pairing

        This uses equation (8) in the paper with an additional normalisation factor
        to achieve proper normalisation behaviour.

        Parameters
        ----------
        trajectory: CompactTrajectory
            The sliced up trajectory to be used to calculate PDFs
        """

        # Calculate histograms for each frame in trajectory
        for partial_value in self.partial_pdfs.values():
            partial_value = np.zeros_like(self.independent_variables['r'])

        for frame in trajectory:
            self._calculate_histogram(frame)

        # Calculate element-independent prefactor
        prefactor = self.universe_volume / (4.0 * np.pi * self.r ** 2 * self.r_step)
        for partial_name, partial_value in self.partial_pdfs.items():
            # Takes the product of the number of atoms of each element in the partial
            atom_number_product = np.multiply(*[self.numbers_of_atoms[elem]
                                                for elem in partial_name])
            # Partials of the same element need to be scaled by 2 so that
            # they tend to 1 as r tends to infinity.
            if len(set(partial_name)) == 1:
                partial_value *= 2
            # Apply weightings & normalise by number of trajectories used
            partial_value *= prefactor / (atom_number_product * len(trajectory))

    def _calculate_total_pdf(self) -> None:
        """
        Calculate the total pdf from the partial pairs
        This function calculates the total PDF in accordance with equation (10)
        This calculation is done in-place
        """
        total_number_of_particles = np.sum(list(self.numbers_of_atoms.values()))
        self._dependent_variables['PDF'] = np.zeros_like(self.r)
        # Calculate proportion and scattering length factors of elements in each pair
        for partial_name, partial_value in self.partial_pdfs.items():
            ci_cj = np.ones_like(self.r)
            bi_bj = np.ones_like(self.r)
            for elem in partial_name:
                # Proportion of elements
                ci_cj *= (self.numbers_of_atoms[elem] / total_number_of_particles)
                # Scattering Lengths/Weights
                bi_bj *= self.weights[elem]

            # Partials of differing elements need to be scaled by 2 when added to total,
            # as only one of the indentical pairs is considered
            # (e.g. for water H-O is added but not O-H)
            if len(set(partial_name)) == 1:
                norm_fac = 1.
            else:
                norm_fac = 2.

            self._dependent_variables['PDF'] += ci_cj * bi_bj * (partial_value - 1) * norm_fac


    def _slice_trajectory(self, trajectory: CompactTrajectory, **settings: dict) \
            -> CompactTrajectory:
        """
        Slice the trajectory into frames used to calculate an average total PDF

        Parameters
        ----------
        trajectory: CompactTrajectory
            The trajectory to slice
        settings: dict
            A dictionary of kwargs used for the pdf calculation
            n_frames : int
                The number of frames from which the pdf and its error are
                calculated. These frames are selected uniformly, and the step is taken to be
                n_frames / total number of frames rounded to the nearest positive integer.
                If this is not passed, 1% of the total number of frames are used
                (rounded up to the nearest positive integer).
        Returns
        -------
        CompactTrajectory
            The slices of the trajectory (selected frames) to use
        """
        # np.max ensures that n_frames is at least 1 (relevant if total_n_frames < 100)
        total_n_frames = len(trajectory)
        n_frames = settings.get('n_frames', np.max([1, total_n_frames // 100]))
        if n_frames < 1 or n_frames > total_n_frames:
            raise ValueError('n_frames must be between 1 and the total number'
                             ' of frames in the trajectory (inclusive)')
        # If only a single frame then set frame_step > total_n_frames
        frame_step = (total_n_frames + 1 if n_frames == 1
                      else ((total_n_frames - 1) // n_frames) + 1)
        return trajectory[0:total_n_frames:frame_step]


    def _parse_apply_MD_settings(self, trajectory: CompactTrajectory, settings: dict) -> None:
        """
        Parses the MD settings and applies them to the class

        This includes setting:
            - the number of evenly spaced frames for which the ``PDF`` will be
              averaged
            - the partial pairs for which ``PDF`` will be calculated
            - the elements involved in the ``PDF``
            - the weights for each element
            - the volume of the universe
            - the independent variables (``r``)
        and slicing the trajectory as requested by the user

        Parameters
        ----------
        trajectory: CompactTrajectory
            A `CompactTrajectory` object to be used for the PDF calculations

        settings : dict
            A `dict` of settings to be parsed

        Raises
        ------
        ValueError
            If ``n_frames`` is less than 1 or greater than the number of frames
            in the ``CompactTrajectory``
        TypeError
            If ``r`` is in settings as well any of ``r_min``, ``r_max``, and
            ``r_step``

        Warnings
        --------
        UserWarning
            If one or two of ``r_min``, ``r_max``, and ``r_step`` have been
            passed, user is warned that three are required to set ``r``.
        """

        # Slice the trajectory in accordance with user config
        self.trajectory = self._slice_trajectory(trajectory, **settings)

        # If no subset is specified, combinations of all elements are used, so
        # that all possible partials will be calculated. The element set is
        # sorted so that partial pair strings will always be ordered
        # alphabetically.
        self.partial_strings = settings.get('subset',
                                            list(combinations_with_replacement(
                                                sorted(trajectory.element_set), 2)))

        # Create element set from elements in partials. The weights are then
        # determined from these.
        self.elements = set(chain.from_iterable(self.partial_strings))
        self.weights = self._set_weights(self.elements,
                                         settings.get('b_coh', {}))
        self.numbers_of_atoms = self._set_numbers(self.elements,
                                                  self.trajectory.element_list)

        self.universe_dimensions = np.array((settings.get('dimensions')
                                             or trajectory.universe.dimensions))
        self.universe_volume = np.prod(self.universe_dimensions)
        # PDF only valid where number of atoms is conserved over trajectories
        self.n_atoms = self.trajectory.n_atoms

        # Create independent_variables dictionary if it doesn't exist
        if not hasattr(self, 'independent_variables'):
            self.independent_variables = ({'r': settings['r']} if 'r' in settings
                                          else {})

        # If rmin, rmax and rstep are in settings, overwrite existing values for
        # independent variable. If one or two are in settings, warn the user
        # that all three are required to set r.
        r_kwargs = ['r_min', 'r_max', 'r_step']
        r_kwarg_defined = any(r_kw in settings.keys() for r_kw in r_kwargs)
        if 'r' in settings:
            self.r = settings.get('r')
            if r_kwarg_defined:
                raise TypeError('r cannot be passed if r_min, r_max or r_step'
                                ' are passed')
        if all(r_kw in settings.keys() for r_kw in r_kwargs):
            self.r = np.arange(settings['r_min'],
                               settings['r_max'] + settings['r_step'],
                               settings['r_step'])
        elif r_kwarg_defined:
            warnings.warn('Setting r requires r_min, r_max and r_step to all be'
                          ' set. Using existing r instead.')
        self.r_step = self.r[1] - self.r[0]

        self.partial_pdfs = {partial_string:
                             np.zeros(
                                 np.shape(self.independent_variables['r']))
                             for partial_string in self.partial_strings}

        self._dependent_variables = {}


    def _calculate_histogram(self, trajectory: CompactTrajectory) -> None:
        """
        Partitions the atomic positions into regions where they are within
        ``r_max`` from all other atoms
        """
        def get_component_lengths(universe_dim: float) -> np.ndarray:
            """
            Use ``r`` values for each component that are at least as big as
            ``r_max``, but that are a factor of the dimensions

            When ``r > r_max``, some atom pairs within each partition will have
            separations which will not be histogrammed (as they are
            ``> r_max``), however this is unavoidable
            """

            r_max = np.max(self.independent_variables['r'])
            return universe_dim / (universe_dim // r_max)

        r_min = np.min(self.r) - self.r_step / 2
        r_max = np.max(self.r) + self.r_step / 2
        _, bin_edges = np.histogram([], len(self.r), range=(r_min, r_max))
        bin_edges_squared = bin_edges**2

        part_comps = np.array(list(map(get_component_lengths,
                                       self.universe_dimensions)))
        partitions = self._partition(trajectory,
                                    part_comps)
        # Get the partition_indexes and the pairs of partitions. As well as
        # calculating atom pairs within a partition, each partition will have 26
        # neighbors for which atom pairs must be calculated.
        partition_indexes = list(self._calculate_partition_indexes(part_comps))
        partition_pair_indexes = self._get_partition_pairs(part_comps)

        # Now we iterate over all the possible combinations of chemical elements
        for partial_string in self.partial_strings:
            elem1, elem2 = partial_string

            # And here we iterate over possible pairs of neighbouring partitions
            for part1, part2 in partition_pair_indexes + [(x,x) for x in partition_indexes]:

                array1 = partitions[elem1][part1]
                array2 = partitions[elem2][part2]

                n_at_array1 = len(array1)
                n_at_array2 = len(array2)
                array1 = array1.reshape(1,n_at_array1,3)
                array2 = array2.reshape(n_at_array2,1,3)
                difference = array1 - array2
                for dim in range(3):
                    temp_array = difference[:,:,dim]
                    box_side = abs(self.universe_dimensions[dim])
                    # Correct for periodic boundary conditions
                    crit1 = np.where(temp_array > 0.5*box_side)
                    crit2 = np.where(temp_array < -0.5*box_side)
                    temp_array[crit1] -= box_side
                    temp_array[crit2] += box_side
                    difference[:,:,dim] = temp_array

                if elem1==elem2 and np.all(part1 == part2):
                    temp = np.sum(difference**2, axis = 2)
                    temp = np.triu(temp, k=1)
                    distance_squared = temp.ravel()
                else:
                    distance_squared = np.sum(difference**2, axis = 2).ravel()

                histogram, _ = np.histogram(distance_squared, bins=bin_edges_squared)

                self.partial_pdfs[partial_string] += histogram

    def _partition(self, trajectory: CompactTrajectory, part_comps: np.ndarray) -> dict:
        """
        Partitions the atomic positions into paritions of dimensions specified
        by ``part_comps``

        ``Atom`` objects are grouped in partitions by ``Atom.element``

        Parameters
        ----------
        positions : numpy.ndarray
            the full CompactTrajectory.positions array, with the
            shape = (num_time_steps, num_atoms, 3)
        element_list : list
            A `list` of `str` with the same length as ``positions``. Each `str`
            specifies the ``Atom.element`` for the corresponding index in
            ``positions``.
        part_comps : numpy.ndarray
            A 3 element array specifying the length of each component for all
            partitions

        Returns
        -------
        dict
            ``{elem:partitions}``, where each ``elem`` is taken from
            ``element_list``, and ``partitions`` is a `dict` of
            ``{index:positions}``, where ``index`` is a `tuple` of 3 `int`
            specifying the index of a partition, and ``positions``
            is an ``array`` of the positions of the atoms which are located
            within the partition.
        """

        # Set up a partitions dictionary separated by element
        partitions = {element: defaultdict(list) for element
                      in self.elements}

        # Add empty lists for all possible partition indexes. This will allow
        # product and combinations to include these indexes (although they
        # will be empty combinations and products)
        for element in self.elements:
            for part_index in self._calculate_partition_indexes(part_comps):
                partitions[element][part_index] = []

        # Drop positions of atoms of elements not included
        filtered_trajectory = trajectory.filter_by_element(self.elements)

        for elem in self.elements:
            subtrajectory = filtered_trajectory.filter_by_element([elem])
            pos_array = subtrajectory.positions[0]
            indices = (pos_array // np.array(part_comps)).astype(int)
            for one_set in np.unique(indices, axis=0):
                temp_set = tuple(one_set)
                new_pos_array = pos_array[np.where(np.all(indices == temp_set, axis=1))]
                partitions[elem][temp_set] = new_pos_array
        return partitions

    def _get_partition_pairs(self, partition_components: np.ndarray) -> list[tuple]:
        """
        Calculates which partitions are neighbours and pairs them. This includes
        partitions that are neighbours due to periodic boundary conditions.

        Parameters
        ----------
        partition_components : numpy.ndarray
            3 element ``array`` of `float`, where each `float` is the length of
            one component of a partition. For example (1., 2., 3.) would mean
            that every component was length 1. in the x direction, 2. in the y
            direction, and 3. in the z direction.

        Returns
        -------
        list of tuples
            ``(parition1, partition2)`` where ``partition1`` and ``partition2``
            are 3 element arrays with the indexes of the partition. Each pair of
            partitions are neighbours.
        """

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

    def _calculate_partition_indexes(self, partition_components: np.ndarray) -> 'product[tuple]':
        return product(*map(np.arange, (self.universe_dimensions
                                        / partition_components).astype('int32'))
                       )

    def _calculate_histogram_from_position_pairs(self, position_pairs: iter) -> np.ndarray:
        """
        Returns a histogram of pair separations calculated from
        ``position_pairs``

        Parameters
        ----------
        position_pairs : iter
            An iterator returning 2 element `tuples`, where each element is a 3
            element vector specifying a position.

        Returns
        -------
        numpy.ndarray
            An histogram of length equal to the length of ``r`` in
            ``independent_variables``, where each count is an atomic separation
            (between the elements in a `tuple` in ``position_pairs``)
        """
        # Use np.histogram to get empty array of correct size and bin edges
        # Assumes constant r step size
        r_min = np.min(self.r) - self.r_step / 2
        r_max = np.max(self.r) + self.r_step / 2
        hist, bin_edges = np.histogram([], len(self.r), range=(r_min, r_max))
        bin_edges_squared = bin_edges**2

        def jit_histogram(separations, bin_edges):
            jit_hist, _ = np.histogram(separations, bins=bin_edges)
            return jit_hist

        # Calculate histograms over blocks. This is mainly for memory management.
        # Numpy also uses blocks, with a BLOCK size of 65536 (256**2),
        # so will use this block size and pad with zeros
        # if position_pairs has been exhausted.
        # pylint: disable=invalid-name
        BLOCK = 65536
        exhausted = False
        while not exhausted:
            # Get next block from iterator
            pair_block = np.fromiter(islice(position_pairs, BLOCK), dtype=object)
            # return histogram if no more exist in the iterator
            if len(pair_block) == 0:
                return hist

            # Unpack & subtract positions from each other
            pos_1 = np.array([tup[0] for tup in pair_block])
            pos_2 = np.array([tup[1] for tup in pair_block])
            difference = np.subtract(pos_1, pos_2)

            if len(pair_block) < BLOCK:
                # Toggle exhausted so that the while loop will stop.
                exhausted = True
                # The calculated differences will be filled with np.array(['inf'])
                # if the iterator ended before reaching the element at position BLOCK (65536).
                padding = np.array([np.array([float('inf'), float('inf'), float('inf')])
                           for _ in range(BLOCK - len(pair_block))])
                difference = np.append(difference, padding, axis=0)

            separations = self._calculate_euclidean_norm_squared(difference)
            hist += jit_histogram(separations, bin_edges_squared)

        return hist

    @staticmethod
    def _calculate_euclidean_norm(vectors: np.ndarray) -> np.ndarray:
        """
        Calculates the Euclidean norm of an array of vectors

        Parameters
        ----------
        vectors : numpy.ndarray
            A 2-D array containing vectors for which the Euclidean norm
            (Euclidean length, L2 distance) is calculated.
        """

        return np.sum(np.square(vectors), 1) ** 0.5

    @staticmethod
    def _calculate_euclidean_norm_squared(vectors: np.ndarray) -> np.ndarray:
        """
        Calculates the Euclidean norm squared of an array of vectors.
        This way we can save time by not calculating the square root,
        since the bin limits of the histogram are also squared.

        Parameters
        ----------
        vectors : numpy.ndarray
            A 2-D array containing vectors for which the Euclidean norm
            (Euclidean length, L2 distance) is calculated.
        """

        return np.sum(np.square(vectors), 1)

    @staticmethod
    def _set_weights(unique_elements: list[str], b_coh: dict) -> dict:
        """
        Sets the weights for each element

        Uses any scattering lengths passed in b_coh, but defaults to values in
        ``MDMC.common.atom_properties``

        Parameters
        ----------
        unique_elements : list of str
            Where each `str` specifies an element
        b_coh : dict
            ``(element:b_c)`` where ``element`` is a `str` specifying an element
            occuring in ``unique_elements``, and ``b_c`` is the weight (coherent
            scattering length) to be set for that element

        Returns
        -------
        dict
            ``(element:weight)`` where ``element`` is a `str` and ``weight`` is
            the corresponding weight
        """

        return {element: b_coh.get(element, B_COH[element]) for element
                in unique_elements}

    @staticmethod
    def _set_numbers(unique_elements: list[str], element_list: list[str]) -> dict:
        """
        Sets the number of atoms of each element

        Parameters
        ----------
        unique_elements : list of str
            Where each `str` specifies an element
        element_list : list of str
            A `list` of the elements for every ``Atom`` in the ``CompactTrajectory``

        Returns
        -------
        dict
            ``(element:number)`` where ``element`` is a `str` and ``number`` is
            the number of atoms of the that element in the ``element_list``
        """

        return {element: element_list.count(element) for element
                in unique_elements}

    @property
    def dependent_variables_structure(self) -> dict[str, list]:
        """
        The shape of the 'PDF' dependent variable in terms of 'r'':
        np.shape(self.PDF)=(np.size(self.r))

        Return
        ------
        dict
            The shape of the PDF dependent variable
        """
        return {'PDF': ['r']}

    @property
    def uniformity_requirements(self) -> dict[str, dict[str, bool]]:
        """
        Defines the current limitations on the atomic separation distance 'r'
        of the ``PairDistributionFunction`` ``Observable.
        The requirement is that 'r' must be uniform, but it does not have to start at zero.

        Return
        ------
        dict[str, dict[str, bool]]
            Dictionary of uniformity restrictions for 'r'.
        """

        return {'r': {'uniform': True, 'zeroed': False}}

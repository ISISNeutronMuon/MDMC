"""Module containing an abstract base class for MD engine facades"""

from abc import ABC, abstractmethod


class MDEngine(ABC):

    """
    Abstract base class for MD engine facades
    """

    def __repr__(self):

        return ('<{0}\n'
                ' {{MD_engine: {MD_engine},\n'
                ' exp_datasets: {exp_datasets},\n'
                ' FoM_calculator: {FoM_calculator},\n'
                ' minimizer: {minimizer},\n'
                ' reset_config: {reset_config},\n'
                ' fit_params: {fit_params},\n'
                ' settings: {settings}}}>').format(self.__class__.__name__,
                                                   **self.__dict__)

    @property
    @abstractmethod
    def saved_config(self):

        """
        Get the saved configuration of the atomic positions

        Returns
        -------
        ``Configuration``
            The atomic positions
        """

        pass

    @abstractmethod
    def setup_universe(self, universe, **settings):

        """
        Creates a ``Universe.configuration`` and populates with
        ``StructuralUnit``

        Parameters
        ----------
        universe : Universe
            A molecular dynamics ``Universe`` which will be setup in the
            ``MDEngine``.
        **settings
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        pass

    @abstractmethod
    def setup_simulation(self, **settings):

        """
        Sets the options required to perform a simulation on a setup
        ``Universe``. Must follow a call to ``setup_universe()``.

        Parameters
        ----------
        universe : Universe
            A molecular dynamics ``Universe`` which will be simulated in the
            ``MDEngine``.
        settings**
            The majority of these are generic but some are specific to the
            ``MDEngine`` that is being used.
        """

        pass

    @abstractmethod
    def minimize(self, n_steps, **settings):

        """
        Minimizes the simulation energy

        Parameters
        ----------
        n_steps : int
            Maximum number of steps for the energy minimization.
        """

        pass

    @abstractmethod
    def run(self, n_steps, equilibration):

        """
        Runs a simulation.  Must follow a call to ``setup_universe()`` and
        ``setup_simulation()``.

        Parameters
        ----------
        n_steps : int
            Number of steps for the time integrator.
        equilibration : bool
            If `True`, run is equilibration which does not store the
            ``trajectory``. Otherwise run is prodution.
        """

        pass

    @abstractmethod
    def convert_trajectory(self, start=0, stop=None, step=1, **settings):

        """
        Parses the trajectory from the ``MDEngine`` format into MDMC format

        Parameters
        ----------
        start : int
            The index of the first trajectory, inclusive.
        stop : int
            The index of the last trajectory, exclusive.
        step : int
            The step size between trajectories.
        **settings
            ``scaled_positions`` (`bool`)
                If the ``trajectory_file`` has scaled ``positions``
            ``atom_IDs`` (`list`)
                LAMMPS ``ID`` of the atoms which should be included. If not passed
                then all atoms are included in the converted trajectory.

        Returns
        -------
        ``Trajectory``
            The ``Trajectory`` from the most recent production simulation
        """

        # convert_trajectory has no range function as it is assumed that the
        # trajectory that is calculated by MD is exactly what is required

        pass

    @abstractmethod
    def update_parameters(self):

        """
        Updates the ``MDEngine`` force field ``Parameter`` objects from the
        ``Universe``
        """

        pass

    @abstractmethod
    def save_config(self):

        """
        Sets ``self.saved_config`` to the current configuration
        """

    @abstractmethod
    def reset_config(self):

        """
        Resets the configuration of the simulation to that in ``saved_config``
        """

        pass

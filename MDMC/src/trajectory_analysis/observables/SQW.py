"""Module for SQw class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 13:24:27"""

import numpy as np

from MDMC.src.trajectory_analysis.observables.obs import Observable

class DynamicStructureFactor(Observable):

    """
    A class for containing, calculating and reading a dynamic structure factor
    """

    @property
    def origin(self):

        return self._origin

    @property
    def data(self):

        return {'independent':self.independent_variables,
            'dependent':self.dependent_variables,
            'errors':self.errors}

    @property
    def independent_variables(self):

        """
        Returns a dictionary of all independent variables
        """

        return self._independent_variables

    @property
    def dependent_variables(self):

        """
        Returns a dictionary of all dependent variables
        """

        return self._dependent_variables

    @property
    def errors(self):

        """
        Returns a dictionary of all errors
        """

        return self._errors

    def read_from_file(self, reader, file_name):

        super(DynamicStructureFactor, self).read_from_file(reader, file_name)
        self._independent_variables = self.reader.independent_variables
        self._dependent_variables = self.reader.dependent_variables
        self._errors = self.reader.errors
        self._origin = 'experiment'

    # TODO: Add neutron weights
    # TODO: Detailed balance correction
    # TODO: Add SQw errors
    def calculate_from_MD(self, MD_input, **params):

        """
        Currently sets all errors to 0 when S(Q,w) is calculated from MD

        Independent variables can either be set previously or defined within
        params
        """
        self.trajectory = MD_input
        self.dt = self.trajectory.times - self.trajectory.times[0]
        self.universe_cell = params.get('cell')

        try:
            self.Q_values = np.array(params.get('Q_values'))
        except KeyError:
            self.Q_values = self.independent_variables['Q']

        if params.get('isotropic', True):
            self.FQt = self._calculate_FQt_orthogonal_Q_vectors()
        else:
            direction = params.get('direction', (1, 0, 0))
            self.FQt = self._calculate_FQt_multiple_Q(direction)

        self.SQw = self._calculate_SQw()
        self.SQw_err = np.zeros(self.SQw.shape)
        self.w = self._change_domain(self.dt)

        self._independent_variables = {'Q':self.Q_values, 'w':self.w}
        self._dependent_variables = {'SQw':self.SQw}
        self._errors = {'SQw':self.SQw_err}
        self._origin = 'MD'

    # TODO: Sum contributions of different directions at rho stage, rather than here
    def _calculate_FQt_orthogonal_Q_vectors(self):

        """
        Calculates Q for three orthgonal directions
        """

        ortho_dir = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        return np.sum([self._calculate_FQt_multiple_Q(dir)
            for dir in ortho_dir], axis = 0)

    def _calculate_FQt_multiple_Q(self, direction):

        """
        Calculates FQt for a range of Q in one direction
        """
        self.Q_vectors = self._calculate_Q_vectors(direction)
        return [self._calculate_FQt_single_Q(Q) for Q in self.Q_vectors]

    # TODO: Refactor to remove horrible indexing [(self.dt == t1)] etc
    # TODO: Consider effects of correlation in consecutive timesteps
    def _calculate_FQt_single_Q(self, Q):

        """
        Calculates intermediate scattering function for a single Q value for all
        time intervals (dt)

        Gets rho for all times for single Q value and then determines
        correlation of rho for every time with every time (including self
        correlation).  Sums correlations with same time interval.  The counter
        normalises FQt depending on how many repetitions of the same time
        interval contributed to that dt.
        """

        rho = self._calculate_rho_single_Q(Q)
        n_atoms = len(self.trajectory.atoms)

        FQt = np.zeros(len(self.dt), dtype = complex)
        counter = np.zeros(len(self.dt))
        for t1 in self.trajectory.times:
            for t2 in self.trajectory.times:
                if t2 >= t1:
                    rho_t1 = rho[(self.dt == t1)]
                    rho_t2 = rho[(self.dt == t2)]
                    corr = self._correlation(rho_t1, rho_t2)
                    FQt[(self.dt == t2 - t1)] += corr
                    counter[(self.dt == t2 - t1)] += 1
        FQt /= (n_atoms * counter)
        return FQt

    # TODO: Implement following method for a single q-shell with random direction q-vectors.
    # This will add resolution effects and remove assumption of isotropy
    def _calculate_rho_single_Q(self, Q):

        """
        Calculates time dependent number density in reciprocal space for a
        single Q value
        """
        rho = []
        for _, time in enumerate(self.trajectory.times):
            rho_temp = [(np.exp(-1j * np.dot(Q, r)))
                for r in self.trajectory[time].positions]
            rho.append(np.sum(rho_temp, axis = 0))
        return np.array(rho)

    # TODO: Replace this with Full Correlation Analysis algorithm?
    def _correlation(self, input1, input2):

        """
        Calculates the correlation between the two inputs
        """

        return np.dot(input1, input2)

    # TODO: Extract out fft calculation into utilities
    def _calculate_SQw(self):

        """
        Calculates SQw from FQt
        """
        SQw = []
        for Ft in self.FQt:
            SQw.append(np.fft.ifft(Ft))

        return np.array(SQw)

    def _change_domain(self, domain):

        """
        Assumes domain of constant step size
        """

        n = domain.size
        step = domain[1] - domain[0]
        return np.pi * np.fft.fftshift(np.fft.fftfreq(domain.size, step))

    def _calculate_Q_vectors(self, direction):

        return np.array([value * np.array(direction)
            for value in self.Q_values])

    def _calculate_SQ(self, trajectory, dir):

        """
        Calculate static structure factor, primarily for testing

        Normalised to number of atoms and number of contributing configurations
        (at different times)
        """

        n_atoms = len(self.trajectory.atoms)
        Q_vectors = self._calculate_Q_vectors(dir)

        self.SQ = []
        for Q in Q_vectors:
            rho_Q = []
            for _, time in enumerate(trajectory.times):
                positions = trajectory[time].positions
                rho_Q_time = np.sum([(np.exp(-1j * np.dot(Q, r)))
                    for r in positions])
                rho_Q.append(rho_Q_time)
            rho_Q = np.sum(rho_Q)
            self.SQ.append(self._correlation(rho_Q, rho_Q))

        self.SQ = np.array(self.SQ) / (n_atoms * len(trajectory.times))

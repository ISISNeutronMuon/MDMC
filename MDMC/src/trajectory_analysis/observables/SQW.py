"""Module for SQW class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 13:24:27"""

import numpy as np
#import uncertainties.unumpy as unp

from MDMC.src.trajectory_analysis.observables.exp_obs import \
    ExperimentalObservable

class DynamicStructureFactor(ExperimentalObservable):

    """
    A class for containing, calculating and reading a dynamic structure factor
    """

    @property
    def from_MD(self):

        NotImplementedError

    @property
    def data(self):

        return self._data

    @property
    def independent_variables(self):

        """
        Assumes there is only a single dependent dataset and its error
        """

        return self.data[0:-2]

    @property
    def dependent_variables(self):

        """
        Returns a single dependent dataset and its error
        """

        return self.data[-2:]

    def read_from_file(self, reader, file_name):

        super(DynamicStructureFactor, self).read_from_file(reader, file_name)
        self._data = self.reader.data

    # TODO: Add neutron weights
    # TODO: Detailed balance correction
    def calculate_from_MD(self, MD_input, **params):

        """
        Currently sets all errors to 0 when S(Q,w) is calculated from MD
        """
        self.trajectory = MD_input
        self.Q_vectors = np.array(params.get('Q_vectors'))
        self.dt = self.trajectory.times - self.trajectory.times[0]
        self.universe_cell = params.get('cell')

        self.FQt = [self._calculate_FQt_single_Q(Q) for Q in self.Q_vectors]
        self.SQw = self._calculate_SQw()
        self.SQw_err = np.zeros(self.SQw.shape)
        self.w = self._change_domain(self.dt)

        self._data = [self.Q_vectors, self.w, self.SQw, self.SQw_err]

    # TODO: Refactor to remove horrible indexing [(self.dt == t1)] etc
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
                    corr = self.correlation(rho_t1, rho_t2)
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
        for time in self.trajectory:
            rho_temp = [(np.exp(-1j * np.dot(Q, r)))
                for r in time.positions]
            rho.append(np.sum(rho_temp, axis = 0))
        return np.array(rho)

    # TODO: Replace this with Full Correlation Analysis algorithm?
    def correlation(self, input1, input2):

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

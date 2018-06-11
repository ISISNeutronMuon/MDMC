"""Module for SQW class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-6 13:24:27"""

import numpy as np
import uncertainties.unumpy as unp

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
        self._data = reader.data

    def calculate_from_MD(self, MD_input, **params):

        """
        Currently sets all errors to 0 when S(Q,w) is calculated from MD
        """
        self.trajectory = MD_input
        self.Q_vectors = params.get('Q_vectors')
        self.dt = self.trajectory.times - self.trajectory.times[0]

        self.FQt = [self._calculate_FQt_single_Q(Q) for Q in self.Q_vectors]

    # TODO: Refactor to remove horrible indexing [(self.dt == t1)] etc
    def _calculate_FQt_single_Q(self, Q):

        """
        Calculates intermediate scattering function for a single Q value for all
        time intervals (dt)

        Gets rho for all times for single Q value and then determines
        correlation of rho for every time with every time (including self
        correlation).  Sums correlations with same time interval.
        """

        rho = self._calculate_rho_single_Q(Q)
        n_atoms = len(self.trajectory.atoms)

        FQt = np.zeros([len(self.dt),2])
        FQt[:,0] = self.dt
        for t1 in self.trajectory.times:
            for t2 in self.trajectory.times:
                if t2 >= t1:
                    rho_t1 = rho[(self.dt == t1)]
                    rho_t2 = rho[(self.dt == t1)]
                    corr = self.correlation(rho_t1, rho_t2)
                    FQt[(self.dt == t2 - t1),1] += corr

        FQt[:,1] /= n_atoms
        return FQt

    # TODO: Implement following method for a single q-shell with random direction q-vectors.
    # This will add resolution effects and remove assumption of isotropy
    def _calculate_rho_single_Q(self, Q):

        """
        Calculates time dependent number density in reciprocal space for a
        single Q value
        """
        rho = np.empty(len(self.trajectory.times))
        for i, time in enumerate(self.trajectory.times):
            rho[i] = sum([(np.exp(-1j * np.dot(Q, r)))
                for r in self.trajectory[time].positions])
        return rho

    # TODO: Replace this with Full Correlation Analysis algorithm?
    def correlation(self, input1, input2=None):

        """
        Calculates the correlation between the two inputs, or the
        autocorrelation for a single input
        """

        return np.dot(input1, input2)

"""Module for RDF class

AUTHOR :    Thomas Farmer        START DATE :    2018-6-5 14:28:35"""

import numpy as np
import uncertainties.unumpy as unp

from MDMC.src.trajectory_analysis.observables.exp_obs import \
    ExperimentalObservable


class RadialDistributionFunction(ExperimentalObservable):

    """
    A class for containing, calculating and reading a radial distribution
    function
    """

    @property
    def from_MD(self):

        return self._from_MD

    @property
    def data(self):
        return self._data
        raise NotImplementedError

    @property
    def dependent_variable(self):
        return self.data['intensity']

    def read_from_file(self, reader, file_name):
        self._from_MD = False
        self._reader = reader
        raw_data = self._reader.read(file_name)
        raise NotImplementedError

        self._data = raw_data

    def calculate_from_MD(self, MD_input, **params):

        # TODO: Test to ensure MD_input (histogram) is correct format

        self._from_MD = True
        self.volume = params['volume']
        self.n_atoms = params['n_atoms']

        self._calculate_prefactor()

        self._data = MD_input['histogram'] * self._prefactor

    def _calculate_prefactor(self):
        raise NotImplementedError
        self._prefactor = None

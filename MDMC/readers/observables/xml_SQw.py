"""XML reader for SQw data"""

import xml.etree.ElementTree as ET

import numpy as np

from MDMC.common import units
from MDMC.common.constants import h_bar
from MDMC.common.decorators import unit_decorator
from MDMC.readers.observables.obs_reader import ObservableReader


class XML_SQw(ObservableReader):

    """
    An XML reader for SQw data
    """

    def parse(self, **settings):

        """
        Parses the xml file

        Currently only parses SQw files

        E is the energy transfer (in ``meV``)
        Q is wavevector transfer (in ``Ang^-1``)
        """

        self._tree = ET.parse(self.file)
        self._root = self._tree.getroot()
        self._root_dict = self.dict_from_element(self._root)

        n_Q = int(self._root_dict['n-q-points'])
        n_w = int(self._root_dict['n-omega-points'])

        Q_unit = units.Unit(self._root_dict['q-unit'])
        w_unit = units.Unit(self._root_dict['omega-unit'])

        # Local variable Q is used for setting self.Q after all children of
        # self._root have been parsed. This is required because a set cannot be
        # passed to self.Q and then the add method called, because the unit
        # decorator converts the set to an a UnitArray, which has no add method.
        # as the
        Q = set()
        w = set()
        self.SQw = []
        self.SQw_err = []

        for child in self._root:
            if child.tag == 'SQomega':
                child_dict = self.dict_from_element(child)
                Q.add(float(child_dict['q']))
                w.add(float(child_dict['omega']))

                # Account for 'no data' in SQw
                SQw = child_dict['S']
                if SQw == 'no data':
                    self.SQw.append(0.)
                    self.SQw_err.append(0.)
                else:
                    self.SQw.append(float(SQw))
                    self.SQw_err.append(float(child_dict['error']))

        # Account for unit conversion after creating the variables
        self.Q = np.sort(np.array(list(Q)))
        self.Q *= Q_unit.conversion_factor / self.Q.unit.conversion_factor

        self.w = np.sort(np.array(list(w)))
        self.w *= w_unit.conversion_factor / self.w.unit.conversion_factor

        self.E = self.w * 1e15 * h_bar

        # the way the Wells Ar data is structured and read in,
        # we need to reshape the self.SQw list with w points
        # in the outer index and Q points in the inner index.
        # we then need to transpose the result to make it consistent
        # with our approach of calculating SQw from MD. The resulting arrays must satisfy:
        # np.shape(SQw) == (np.size(Q), np.size(E))
        self.SQw = np.transpose(np.reshape(np.array(self.SQw), [n_w, n_Q]))
        self.SQw_err = np.transpose(np.reshape(np.array(self.SQw_err), [n_w, n_Q]))

    @property
    def independent_variables(self):

        """
        Get the independent variables, Q (in ``Ang^-1``) and E (``meV``)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return {"Q":self.Q, "E":self.E}

    @property
    def dependent_variables(self):

        """
        Get the dependent variables, SQw (in ``arb``)

        Returns
        -------
        dict
            The dependent variables, SQw (in ``arb``)
        """

        return {"SQw": [self.SQw]}

    @property
    def errors(self):

        """
        Get the errors on the dependent variables

        Returns
        -------
        dict
            The error on SQw (in ``arb``)
        """

        return {"SQw": [self.SQw_err]}

    @property
    def w(self):

        """
        Get or set the energy transfer expressed in angular frequency, w, in
        ``1 / ps``

        Returns
        -------
        array
            Energy transfer as angular frequency, w, in ``1 / ps``
        """

        return self._w

    @w.setter
    @unit_decorator(unit=units.Unit('ps') ** -1)
    def w(self, value):

        self._w = value

    @property
    def E(self):

        """
        Get or set the energy transfer, E, in ``meV``

        Returns
        -------
        array
            Energy transfer, E, in ``meV``
        """

        return self._E

    @E.setter
    @unit_decorator(unit=units.ENERGY_TRANSFER)
    def E(self, value):

        self._E = value

    @property
    def Q(self):

        """
        Get or set the momentum transfer, Q, in ``Ang^-1``

        Returns
        -------
        array
            Momentum transfer, Q, in ``Ang^-1``
        """

        return self._Q

    @Q.setter
    @unit_decorator(unit=units.LENGTH ** -1)
    def Q(self, value):

        self._Q = value

    def dict_from_element(self, element):

        """
        Creates a dictionary from an XML element

        Parameters
        ----------
        element : Element
            An XML element. Must have items method, which must return a list of
            2 element tuples.

        Returns
        -------
        dict
            For each tuple from the xml Element, The first index is the key and
            the second element is the value.
        """

        return {item[0]:item[1] for item in element.items()}

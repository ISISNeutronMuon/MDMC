"""XML reader for SQw data"""

import numpy as np
import xml.etree.ElementTree as ET

from MDMC.common import units
from MDMC.common.constants import h_bar
from MDMC.common.decorators import unit_decorator
from MDMC.readers.readers import Reader


class XML_SQw(Reader):

    """
    An XML reader for SQw data
    """

    def parse(self):

        """
        Parses the xml file

        Currently only parses SQw files

        E is the energy transfer (in meV)
        Q is wavevector transfer (in Ang^-1)
        """

        self._tree = ET.parse(self.file)
        self._root = self._tree.getroot()
        self._root_dict = self.dict_from_element(self._root)
        n_Q = int(self._root_dict['n-q-points'])
        n_w = int(self._root_dict['n-omega-points'])

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

        self.Q = np.sort(np.array(list(Q)))
        self.w = np.sort(np.array(list(w)))
        self.E = self.w * 1e15 * h_bar
        self.SQw = np.reshape(np.array(self.SQw), [n_w, n_Q])
        self.SQw_err = np.reshape(np.array(self.SQw_err), [n_w, n_Q])

    @property
    def independent_variables(self):

        """
        Get the independent variables, Q (in Ang^-1) and E (meV)

        Returns
        -------
        dict
            The independent variables Q and E
        """

        return {"Q":self.Q, "E":self.E}

    @property
    def dependent_variables(self):

        """
        Get the dependent variables, SQw (in arb)

        Returns
        -------
        dict
            The dependent variables, SQw (in arb)
        """

        return {"SQw":self.SQw}

    @property
    def errors(self):

        """
        Get the errors on the dependent variables

        Returns
        -------
        dict
            The error on SQw (in arb)
        """

        return {"SQw":self.SQw_err}

    @property
    def E(self):

        """
        Get or set the energy transfer, E, in meV

        Returns
        -------
        array
            Energy transfer, E, in meV
        """

        return self._E

    @E.setter
    @unit_decorator(unit=units.ENERGY_TRANSFER)
    def E(self, value):

        self._E = value

    @property
    def Q(self):

        """
        Get or set the momentum transfer, Q, in Ang^-1

        Returns
        -------
        array
            Momentum transfer, Q, in Ang^-1
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

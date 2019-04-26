"""Generic xml reader

As this reader is generic, it is incapable of determining the type of dependent
and independent variables it is returning.

AUTHOR :    Thomas Farmer        START DATE :    26/07/2018, 13:42:40"""

import numpy as np
import xml.etree.ElementTree as ET

from MDMC.common import units
from MDMC.common.constants import h_bar
from MDMC.common.decorators import unit_decorator
from MDMC.readers.readers import Reader


class XML_SQw(Reader):

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
        n_Q = float(self._root_dict['n-q-points'])
        n_w = float(self._root_dict['n-omega-points'])

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
        A dictionary containing Q (in Ang^-1) and E (meV)
        """

        return {"Q":self.Q, "E":self.E}

    @property
    def dependent_variables(self):

        """
        A dictionary containing SQw (in arb)
        """

        return {"SQw":self.SQw}

    @property
    def errors(self):

        """
        A dictionary containing the error associated with SQw (in arb)
        """

        return {"SQw":self.SQw_err}

    @property
    def E(self):

        """
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
        Momentum transfer, Q, in Ang^-1
        """

        return self._Q

    @Q.setter
    @unit_decorator(unit=units.LENGTH ** -1)
    def Q(self, value):

        self._Q = value

    def dict_from_element(self, element):

        return {item[0]:item[1] for item in element.items()}

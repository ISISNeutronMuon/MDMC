"""Generic xml reader

As this reader is generic, it is incapable of determining the type of dependent
and independent variables it is returning.

AUTHOR :    Thomas Farmer        START DATE :    26/07/2018, 13:42:40"""

import numpy as np
import xml.etree.ElementTree as ET

from MDMC.readers.readers import Reader


class XML_SQw(Reader):

    def parse(self):

        """
        Parses the xml file

        Currently only parses SQw files
        """

        self._tree = ET.parse(self.file)
        self._root = self._tree.getroot()
        self._root_dict = self.dict_from_element(self._root)
        n_Q = float(self._root_dict['n-q-points'])
        n_w = float(self._root_dict['n-omega-points'])

        self.Q = set()
        self.w = set()
        self.SQw = []
        self.SQw_err = []

        for child in self._root:
            if child.tag == 'SQomega':
                child_dict = self.dict_from_element(child)
                self.Q.add(float(child_dict['q']))
                self.w.add(float(child_dict['omega']))

                # Account for 'no data' in SQw
                SQw = child_dict['S']
                if SQw == 'no data':
                    self.SQw.append(0.)
                    self.SQw_err.append(0.)
                else:
                    self.SQw.append(float(SQw))
                    self.SQw_err.append(float(child_dict['error']))

        self.Q = np.sort(np.array(list(self.Q)))
        self.w = np.sort(np.array(list(self.w)))
        self.SQw = np.reshape(np.array(self.SQw), [n_w, n_Q])
        self.SQw_err = np.reshape(np.array(self.SQw_err), [n_w, n_Q])

    @property
    def independent_variables(self):

        """
        A dictionary containing Q and E
        """

        return {"Q":self.Q, "w":self.w}

    @property
    def dependent_variables(self):

        """
        A dictionary containing SQw
        """

        return {"SQw":self.SQw}

    @property
    def errors(self):

        """
        A dictionary containing the error associated with SQw
        """

        return {"SQw":self.SQw_err}

    def dict_from_element(self, element):

        return {item[0]:item[1] for item in element.items()}

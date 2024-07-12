"""
XML reader for SQw data.
"""
import logging
import xml.etree.ElementTree as ET

import numpy as np

from MDMC.common import units
from MDMC.common.constants import h_bar
from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)


class XML_SQw(SQwReader):
    """
    An XML reader for SQw data.

    Attributes
    ----------
    SQw : ~numpy.ndarray, size(Q) x size(E)
        2D array of intensity of ``S``
    SQw_err : ~numpy.ndarray, size(Q) x size(E)
        2D array of error in ``S``
    Q : ~numpy.ndarray
        1D array of wavevector transfer (in ``Ang^-1``).
    w : ~numpy.ndarray
        1D array of frequency (in ``ps^-1``).
    E : ~numpy.ndarray
        1D array of energy transfer (in ``meV``).
    """

    def parse(self, **settings: dict) -> None:
        """
        Parse the .xml file.

        Currently only parses SQw files.

        Parameters
        ----------
        **settings : dict
            Extra options.
        """

        _tree = ET.parse(self.file)
        _root = _tree.getroot()
        _root_dict = self.dict_from_element(_root)

        n_Q = int(_root_dict['n-q-points'])
        n_w = int(_root_dict['n-omega-points'])

        Q_unit = units.Unit(_root_dict['q-unit'])
        w_unit = units.Unit(_root_dict['omega-unit'])

        # Local variable Q is used for setting self.Q after all children of
        # self._root have been parsed. This is required because a set cannot be
        # passed to self.Q and then the add method called, because the unit
        # decorator converts the set to an a UnitArray, which has no add method.
        # as the
        Q = set()
        w = set()
        self.SQw = []
        self.SQw_err = []

        for child in _root:
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
        self.SQw_err = np.transpose(np.reshape(
            np.array(self.SQw_err), [n_w, n_Q]))

        # Change any zero error points to
        # inf so that error calculations can still be performed on them.
        if np.any(self.SQw_err <= 0.):
            self.SQw_err[np.where(self.SQw_err <= 0.)] = float('inf')
            logger.warning(self.SQW_ERR_WARNING)

    @staticmethod
    def dict_from_element(element: ET.Element) -> dict:
        """
        Create a dictionary from an XML element.

        Parameters
        ----------
        element : ~xml.etree.ElementTree.Element
            An XML element. Must have ``items`` method, which must
            return a list of 2 element tuples.

        Returns
        -------
        dict
            For each tuple from the xml Element, The first index is the key and
            the second element is the value.
        """
        return {item[0]: item[1] for item in element.items()}

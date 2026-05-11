# MDMC is a package for the optimisation of classical potentials with experimental data
# Copyright (C) 2026 MDMC Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""XML reader for SQw data"""

import logging
import xml.etree.ElementTree as ET
from typing import Any, TextIO

import numpy as np

from MDMC.common import units
from MDMC.common.constants import h_bar
from MDMC.readers.observables.obs_reader import SQwReader

logger = logging.getLogger(__name__)


class XML_SQw(SQwReader):
    """
    An XML reader for SQw data
    """

    def parse(self, **settings: Any) -> None:
        """
        Parses the xml file

        Currently only parses SQw files

        E is the energy transfer (in ``meV``)
        Q is wavevector transfer (in ``Ang^-1``)
        """

        _tree = ET.parse(self.file)
        _root = _tree.getroot()
        _root_dict = self.dict_from_element(_root)

        n_Q = int(_root_dict["n-q-points"])
        n_w = int(_root_dict["n-omega-points"])

        Q_unit = units.Unit(_root_dict["q-unit"])
        w_unit = units.Unit(_root_dict["omega-unit"])

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
            if child.tag == "SQomega":
                child_dict = self.dict_from_element(child)
                Q.add(float(child_dict["q"]))
                w.add(float(child_dict["omega"]))

                # Account for 'no data' in SQw
                SQw = child_dict["S"]
                if SQw == "no data":
                    self.SQw.append(0.0)
                    self.SQw_err.append(0.0)
                else:
                    self.SQw.append(float(SQw))
                    self.SQw_err.append(float(child_dict["error"]))

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

        # Change any zero error points to
        # inf so that error calculations can still be performed on them.
        if np.any(self.SQw_err <= 0.0):
            self.SQw_err[np.where(self.SQw_err <= 0.0)] = float("inf")
            msg = "\n We have set the error bar to infinity for any zero error values, this allows \
                us to calculate chi-squared but effectively ignores these points, this may not \
                be what you want to do, consider using a FoM which doesn't need errors i f\
                this is an issue \n"
            logger.error(msg)

    @staticmethod
    def dict_from_element(element: TextIO) -> dict:
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

        return {item[0]: item[1] for item in element.items()}

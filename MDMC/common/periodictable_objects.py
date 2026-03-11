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

""" Utility methods for dealing with periodictable atom_objects """

import periodictable


def create_list_of_element_objects(old_elements: list) -> list:
    """
    A list of strings of elements/isotopes is converted into a list of periodictable element/isotope
    objects. Different forms of how an atom can be specified is taken into account. The forms
    allowed here are (using Argon as an example) 'Ar', 'Ar[36]', '36-Ar'.

    Parameters
    ----------

    old_elements: list
        a list containing strings of elements/isotopes.

    Returns
    -------

    elements_list: list
        list of periodictable element/isotope objects
    """

    elements_list = []
    for element in old_elements:
        if '-' in element:
            new_element = element.split('-')
            atom_object = periodictable.elements.symbol(new_element[1])[int(new_element[0])]
        elif '[' in element:
            new_element = element.split('[')
            # remove closed bracket from string
            new_element[1] = new_element[1][:-1]
            atom_object = periodictable.elements.symbol(new_element[0])[int(new_element[1])]
        else:
            atom_object = periodictable.elements.symbol(element)
        elements_list.append(atom_object)
    return elements_list

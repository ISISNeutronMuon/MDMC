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
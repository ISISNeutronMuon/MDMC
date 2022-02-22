"""Structural unit utility functions"""

def is_atom(atom: object) -> bool:
    """
    Checks if the passed object is an instance of the ``Atom`` class.
    Parameters
    ----------
    atom
        Object to checked
    Returns
    -------
    bool
        Boolean if the passed Object is an ``Atom`` or not.
    """
    # pylint: disable=import-outside-toplevel
    # to avoid circular importing
    from MDMC.MD.structural_units import Atom
    return isinstance(atom, Atom)

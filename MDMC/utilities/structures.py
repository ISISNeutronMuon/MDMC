"""
Utility functions for structures.
"""


def is_atom(atom: object) -> bool:
    """
    Checks if the passed object is an instance of the ``Atom`` class.

    Parameters
    ----------
    atom
        Object to be checked.

    Returns
    -------
    bool
        `True` if the passed Object is an ``Atom``.
    """
    # pylint: disable=import-outside-toplevel, cyclic-import
    # we are importing here on purpose to avoid circular importing
    from MDMC.MD.structures import Atom
    return isinstance(atom, Atom)

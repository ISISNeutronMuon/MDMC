
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
    from MDMC.MD.structural_units import Atom
    return isinstance(atom, Atom)

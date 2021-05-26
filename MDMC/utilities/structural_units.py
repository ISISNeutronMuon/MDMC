
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

def parse_structural_unit_IDs(structural_units):
    """
    Converts all ``int`` elements in ``atoms`` into ``StructuralUnit`` objects with that ``int`` as
    their ``ID``. Any elements that are not ``int`` (i.e. any that are already a
    ``StructuralUnit``) are not affected.

    Parameters
    ----------
    structural_units : list
        A `list` of ``StructuralUnit`` or ``int`` corresponding to an ``StructuralUnit.ID``

    Returns
    -------
    list
        ``StructuralUnit`` objects

    Raises
    ------
    KeyError
        If one of the ``int`` in ``structural_units`` does not correspond to an existing
        ``StructuralUnit.ID``.
    """

    parsed_units = []
    for unit in structural_units:
        if isinstance(unit, int):
            try:
                parsed_unit = StructuralUnit._ID_dict[unit]
            except KeyError as error:
                msg = 'No atom found with ID {}'.format(unit)
                raise KeyError(msg) from error
        else:
            parsed_unit = unit
        parsed_units.append(parsed_unit)

    return parsed_units

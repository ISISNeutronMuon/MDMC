
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
    from MDMC.MD.structures import Atom
    return isinstance(atom, Atom)

def parse_structure_IDs(structures):
    """
    Converts all ``int`` elements in ``atoms`` into ``StructuralUnit`` objects with that ``int`` as
    their ``ID``. Any elements that are not ``int`` (i.e. any that are already a
    ``StructuralUnit``) are not affected.

    Parameters
    ----------
    structures : list
        A `list` of ``StructuralUnit`` or ``int`` corresponding to an ``StructuralUnit.ID``

    Returns
    -------
    list
        ``StructuralUnit`` objects

    Raises
    ------
    KeyError
        If one of the ``int`` in ``structures`` does not correspond to an existing
        ``StructuralUnit.ID``.
    """

    parsed_units = []
    for unit in structures:
        if isinstance(unit, int):
            try:
                from MDMC.MD.structures import StructuralUnit
                parsed_unit = StructuralUnit._ID_dict[unit]
            except KeyError as error:
                msg = 'No atom found with ID {}'.format(unit)
                raise KeyError(msg) from error
        else:
            parsed_unit = unit
        parsed_units.append(parsed_unit)

    return parsed_units

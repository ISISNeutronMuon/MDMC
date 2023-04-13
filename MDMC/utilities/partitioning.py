"""Partitioning utilities

Utilities related to partitioning iterators into their composite components

Notes
-----

"""

##
from itertools import tee


def partition(items, predicate):

    """
    Partitions an ``iterable`` using a predicate

    Parameters
    ----------
    items : iterable
        An ``interable`` to be partitioned.
    predicate : function
        A predicate that can be applied to items to returned `True` or `False`.

    Returns
    -------
    `tuple`
        A `tuple` of (``gen_true``, ``gen_false``), where ``gen_true`` is a
        generator of all items for which the ``predicate`` returned `True`, and
        ``gen_false is a generator of all items for which the ``predicate``
        returned `False`
    """

    iter_a, iter_b = tee((predicate(item), item) for item in items)
    return ((item for pred, item in iter_a if pred),
            (item for pred, item in iter_b if not pred))


def partition_interactions(interactions, names, unpartitioned=False, lst=False):

    """
    Partitions an ``iterable`` of ``Interaction`` objects using a `list` of
    ``Interaction`` ``names``

    This occurs by using ``partition`` to filter out one ``Interaction`` type
    for each loop, so previously identified ``Interactions`` are no longer
    considered.

    Parameters
    ----------
    interactions : iterable of Interactions
        An ``interable`` of ``Interaction`` objects to be partitioned.
    names : list of str
        Names of ``Interaction`` classes.
    unpartitioned : bool, optional
        If `True`, then a generator containing any ``Interaction`` objects that
        did not have a name in ``names`` is returned as an additional item in
        the `tuple`. Default is `False`.
    lst : bool, optional
        If `True`, then the returned `tuple` contains `list` rather than
        generators. ``Interaction`` objects which have the name specified by
        ``names[n]``. Default is `False`.

    Returns
    -------
    `tuple`
        A `tuple` of length ``len(names)`` where ``index`` ``n`` is a generator
        of all of the ``Interaction`` objects which have the name specified by
        ``names[n]``. If ``unpartitioned`` is `True`, `tuple` is length ``n+1``.
        If ``lst`` is `True`, the generators are replaced by `list`.

    Example
    -------
    Partion ``interactions`` into ``Bonds`` and ``BondAngles``:

        .. highlight:: python
        .. code-block:: python

            bonds, angles = partition_interactions(interactions,
                                                   ['Bond, BondAngle'])
    """

    interaction_lst = [None] * len(names)
    for i, name in enumerate(names):
        interaction_lst[i], interactions = partition(interactions, lambda x, n=name: x.name == n)

    if unpartitioned:
        interaction_lst += [interactions]
    if lst:
        interaction_lst = [list(i) for i in interaction_lst]
    return tuple(interaction_lst)

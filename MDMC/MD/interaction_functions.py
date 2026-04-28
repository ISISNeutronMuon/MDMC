"""A module for storing atomic interaction functions

Contains class InteractionFunction from which all interaction function classes
must derive.  All functions describing atomic interactions must be added to this
module in order to be called by a universe.  If needed, the interaction function
classes can be extended to contain actual function definitions.

Contains class Parameter, which defines the name and value of each
parameter which belongs to an InteractionFunction, and whether the parameter is
fixed, has constraints or is tied.

Contains filters for filtering list of parameters based on a predicate."""

from __future__ import annotations

import functools
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

import numpy as np

from MDMC.common import units
from MDMC.common.decorators import repr_decorator
from MDMC.MD.parameters import Parameter, Parameters

if TYPE_CHECKING:
    from collections.abc import Callable

    from MDMC.MD.interactions import Interaction


@repr_decorator("parameters")
class InteractionFunction:
    """
    Base class for interaction functions, which can be user supplied

    Parameters
    ---------
    val_dict : dict
        ``name:value`` pairs. Currently this must be ordered alphabetically.
        ``value`` must either be an object with a ``value`` and a ``unit`` (e.g.
        a ``UnitFloat`` object), or a (`float`, `str`) `tuple`, where `float` is
        the value and `str` is the unit.
    """

    def __init__(self, val_dict: dict):

        # locals which are excluded from Parameter creation
        excluded = ["self", "settings", "__class__"]
        parameters = Parameters()
        for name, value in val_dict.items():
            if name not in excluded:
                parameter = Parameter(value, name)
                parameters.append(parameter)
                # Create an attribute with the same name as the Parameter
                setattr(self, parameter.type, parameter)
        self.parameters = parameters

    def __str__(self) -> str:

        parameters = " ".join(
            [p.name + ": " + str(p.value) + "," for p in self.parameters.values()],
        ).strip(",")
        return f"{self.__class__.__name__} {parameters}"

    def __eq__(self, other):

        return id(self) == id(other) or all(
            [
                (type(self) is type(other)),
                (
                    [p.type for p in self.parameters.values()]
                    == [p.type for p in other.parameters.values()]
                ),
                (self.parameters_values == other.parameters_values),
            ],
        )

    @property
    def parameters(self) -> Parameters:
        """
        Get or set the interaction function's parameters

        Returns
        -------
        Parameters
            A Parameters object containing each ``Parameter``
        """

        return self._parameters

    @parameters.setter
    def parameters(self, value: Parameters):

        self._parameters = value

    @property
    def parameters_values(self) -> np.ndarray:
        """
        Get the values for all ``Parameters`` objects

        Returns
        -------
        numpy.ndarray
            A NumPy ``array`` of values for all ``Parameter``
        """

        return np.array([p.value for p in self.parameters.values()])

    @property
    def name(self) -> str:
        """
        Get the name of the class of the ``InteractionFunction``

        Returns
        -------
        str
            The class name
        """

        return self.__class__.__name__

    def set_parameters_interactions(self, interaction: Interaction) -> None:
        """
        Sets the ``parent`` ``Interaction`` for all ``Parameters`` objects

        Parameters
        ----------
        interaction : Interaction
            An ``Interaction`` to set as the ``parent`` of all the
            ``Parameters``
        """

        for parameter in self.parameters:
            self.parameters[parameter].interactions = interaction


def inter_func_decorator(*parameter_units) -> Callable:
    """
    Decorates a method to add units to all positional and any relevant keyword
    arguments

    Designed for adding units to parameters of ``__init__`` method for
    subclasses of ``InteractionFunction``.

    Parameters
    ----------
    *parameter_units : tuple
        One or more `tuple` where the first element is a `str` giving the
        keyword name of an expected parameter. The second element is a
        `str` or ``Unit``, where each `str` (or ``Unit``) is a unit
        which is applied to the corresponding value passed to the decorated
        method. If one of the values is unitless, pass `None` at the
        corresponding index in ``parameter_units``.

    Examples
    --------
    The following adds units of ``'Ang'`` to parameter ``alpha``, units of
    ``'s'`` to the parameter ``beta``, and units of ``'atm'`` to the parameter
    ``gamma``:

        .. highlight:: python
        .. code-block:: python

            @inter_func_decorator(('alpha', 'Ang'), ('beta', 's'), ('gamma', 'Pa'))
            def __init__(self, alpha, beta, gamma):
                ...

    If one of the parameters is unitless, this can be set with `None` (in which
    case the returned type will be the same as the original value i.e. a
    ``UnitFloat`` or ``UnitNDArray`` will not be created). So to set ``epsilon``
    as unitless:

        .. highlight:: python
        .. code-block:: python

            @inter_func_decorator(('delta', 'arb'), ('epsilon', None), ('gamma', 'deg'))
            def __init__(self, delta, epsilon, gamma):
                ...
    """

    # Ignore pylint warning for decorator inner function docstrings
    # pylint: disable=missing-docstring
    def decorator(func):
        def unit_creator(value, unit):
            # If no unit is provided, assume unitless and just return value
            if unit is None:
                return value
            # try/except to determine whether value is float or array
            try:
                return units.UnitFloat(value, unit)
            except TypeError:
                return units.unit_array(value, unit)

        @functools.wraps(func)
        def wrapper(self, *values, **settings):
            # If the name associated with a unit in parameter_units matches a
            # keyword argument (in **settings), then associate that parameter
            # with its unit. If not, then the parameter has been passed as a
            # positional argument (in *values) and so we associate the next
            # entry in *values with that unit (as we can rely on the order of
            # the arguments being the same as the order of units in the
            # decorator)
            values = list(values)
            for name, unit in parameter_units:
                if name in settings:
                    settings[name] = unit_creator(settings[name], unit)
                else:
                    settings[name] = unit_creator(values.pop(0), unit)

            return func(self, **settings)

        return wrapper

    return decorator


class Buckingham(InteractionFunction):
    r"""
    The Buckingham potential (in units of ``kJ mol^-1``) for the interaction of
    2 atoms at distance r (in ``Ang``) has the form:

    .. math::

        {\Phi _{12}(r)=A\exp \left(-Br\right)-{\frac {C}{r^{6}}}}

    The first and second terms represent a repulsion and attraction
    respectively, and so all parameters :math:`A`, :math:`B` and :math:`C`
    should be positive in order to be physically valid.

    Parameters
    ----------
    A : float
        The Buckingham parameter A in units of ``kJ mol^-1``
    B : float
        The Buckingham parameter B in units of ``Ang^-1``
    C : float
        The Buckingham parameter C in units of ``Ang^6 kJ mol^-1``

    Examples
    --------
    The following creates a ``Dispersion`` interaction with a ``Buckingham``
    functional form:

    .. highlight:: python
    .. code-block:: python

        buck = Buckingham(3.65e-18, 6.71, 6.94e-22)
        O_disp = Dispersion(universe, (O.atom_type, O.atom_type), function=buck)
    """

    @inter_func_decorator(
        ("A", units.ENERGY),
        ("B", units.LENGTH**-1),
        ("C", units.LENGTH**6 * units.ENERGY),
    )
    def __init__(self, A: float, B: float, C: float):

        super().__init__(locals())


class Coulomb(InteractionFunction):
    r"""
    Coulomb interaction for charged particles:

    .. math::

        E = \frac{Cq_{i}q_{j}}{r}

    Parameters
    ----------
    charge : float
        The charge in units of ``e``

    Examples
    --------
    The following creates a ``Coulombic`` interaction with a ``Coulomb``
    functional form:

    .. highlight:: python
    .. code-block:: python

        O = Atom('O')
        coul = Coulomb(-0.8476)
        O_coulombic = Coulombic(atoms=O, cutoff=10., function=coul)

    As ``Coulomb`` is the default functional form of ``Coulombic`` interactions
    when an ``Atom`` is created, this is equivalent to setting the ``charge``
    on an ``Atom``:

    .. highlight:: python
    .. code-block:: python

        O = Atom('O', charge=-0.8476)
    """

    @inter_func_decorator(("charge", units.CHARGE))
    def __init__(self, charge: float):

        super().__init__(locals())


class HarmonicPotential(InteractionFunction):
    r"""
    Harmonic potential for bond stretching, and angular and improper dihedral
    vibration, with the form:

    .. math::

        E = K(r-r_0)^2

    As ``HarmonicPotential`` can be used with several different ``Interaction``
    types, the ``Interaction`` type must be specified, so that the correct units
    can be assigned to the ``equilibrium_state`` and ``potential_strength``
    parameters.

    Parameters
    ----------
    equilibrium_state : float
        The equilibrium state of the object in either ``Ang`` or ``degrees``,
        depending on the ``interaction_type`` passed.
    potential_strength : float
        The potential strength in units of ``kJ mol^-1 Ang^-2`` (linear) or
        ``kJ mol^-1 rad^-2`` (angular/improper), depending on the
        ``interaction_type`` passed.
    **settings
        interaction_type : str
            A `str` specifying either ``'bond'``, ``'angle'`` or ``'improper'``.
            This assigns the correct units to ``equilibrium_state`` and
            ``potential_strength``. This keyword must be passed.

    Raises
    ------
    ValueError
        The ``interaction_type`` must be ``'bond'``, ``'angle'``, or
        ``'improper'``
    TypeError
        An ``interaction_type`` of ``'bond'``, ``'angle'``, or ``'improper'``
        must be passed

    Examples
    --------
    The following creates a ``HarmonicPotential`` for a ``Bond`` interaction:

    .. highlight:: python
    .. code-block:: python

        hp = HarmonicPotential(1., 4637., interaction_type='bond')

    The following creates a ``HarmonicPotential`` for a ``BondAngle``
    interaction:

    .. highlight:: python
    .. code-block:: python

        hp = HarmonicPotential(109.47, 383., interaction_type='angle')

    The following creates a ``HarmonicPotential`` for a ``DihedralAngle``
    interaction with ``improper==True`` (i.e. an improper dihedral):

    .. highlight:: python
    .. code-block:: python

        hp = HarmonicPotential(180., 20.92, interaction_type='improper')
    """

    def __new__(cls, equilibrium_state: float, potential_strength: float, **settings: Any):

        # interaction_type is a required keyword, but has to be passed through
        # settings so that it can be correctly passed in inter_func_decorator
        try:
            if settings["interaction_type"].lower() == "bond":
                eq_unit = units.LENGTH
                pot_unit = units.ENERGY / units.LENGTH**2
            elif settings["interaction_type"].lower() in ("angle", "bondangle", "improper"):
                eq_unit = units.ANGLE
                # Use radians rather than system angle units (degrees)
                pot_unit = units.ENERGY / units.Unit("rad") ** 2
            else:
                raise ValueError('The interaction_type must be "bond", "angle", or "improper"')
        except KeyError as err:
            raise TypeError(
                'An interaction_type of "bond", "angle" or improper" must be passed',
            ) from err
        # Create a new (uninitialized) HarmonicPotential
        h_pot = super().__new__(cls)
        # Decorate __init__ with inter_func_decorator to apply to correct units
        # (which are dependent on the interaction_type) to the equilibrium_state
        # and potential_strength parameters. This decoration has to be done in
        # new rather than directly on __init__ as it requires the dynamically
        # defined interaction_type to determine the units. cls_init (which is a
        # copy of the original cls.__init__) is required (rather than just
        # applying the decorator directly to cls.__init__) because after the
        # first HarmonicPotential is called, the cls.__init__ will be decorated.
        # Therefore, after this if cls.__init__ is decorated again, it will be
        # irrelevant, as the original decorator will be the last called. So
        # cls._init is used as it will always be equivalent to the undecorated
        # __init__ visible below.
        cls.__init__ = inter_func_decorator(
            ("equilibrium_state", eq_unit),
            ("potential_strength", pot_unit),
        )(cls._init)
        return h_pot

    def __init__(self, equilibrium_state, potential_strength, **settings):

        super().__init__(locals())

    # Declare a class method equal to the __init__ method
    _init = __init__


class Periodic(InteractionFunction):
    r"""
    Periodic potential for proper and improper dihedral interactions, with the
    form:

    .. math::

        E = \sum_{i=1,m}K_i[1.0+\cos(n_i\phi-d_i)]

    where \phi is angle between the ijk and jkl planes (where i, j, k, and l are
    the four atoms of the dihedral).

    Parameters
    ----------
    K1 : float
        The K parameter (energy) of the first order term, in units of
        ``kJ mol^-1``
    n1 : int
        The n parameter of the first order term, which is unitless. This must be
        a non-negative `int`.
    d1 : float
        The d parameter (angle) of the first order term, in units of ``deg``
    *parameters
        K, n, and d parameters for higher order terms. These must be ordered K2,
        n2, d2, K3, n3, d3, K4, n4, d4 etc. The types and units of these
        parameters are the same as the corresponding first order parameters
        listed above.

    Examples
    --------
    The following creates a first order ``Periodic`` for a ``DihedralAngle``
    interaction with ``improper==True`` (i.e. an improper dihedral):

    .. highlight:: python
    .. code-block:: python

        periodic = Periodic(87.864, 2, 180.)
        improper = DihedralAngle(atoms=[C, H1, H2, O], improper=True,
                                 function=periodic)

    The following creates a third order ``Periodic`` for a ``DihedralAngle``
    interaction with ``improper==False`` (i.e. a proper dihedral), with
    ``K1=3.53548``, ``K2=-4.02501`` and ``K3=2.98319``:

    .. highlight:: python
    .. code-block:: python

        periodic = Periodic(3.53548, 1, 0., -4.02501, 2, 180., 2.98319, 3, 0.)
        proper = DihedralAngle(atoms=[N, C1, C2, C3], improper=False,
                               function=periodic)
    """

    def __init__(self, K1: float, n1: float, d1: float, *parameters: float):

        # Check that total number of parameters is divisible by 3
        # Check that all n values are non-negative ints
        val_dict = {}
        all_parameters = [iter((K1, n1, d1) + parameters)] * 3
        # Assign attributes K$, n$, d$, where $ is the order (e.g. K1, n1, d1
        # for the first order etc.)
        for order, order_parameters in enumerate(zip_longest(*all_parameters), start=1):
            # If *parameters is not divisible by three, one or two indexes of
            # order_parameters will be None
            if None in order_parameters:
                raise TypeError(
                    "*parameters must contain a K, n, and d value for"
                    " each order >2 (i.e. it should contain a"
                    " number of values exactly divisible by 3)",
                )
            if not isinstance(order_parameters[1], (int, np.integer)):
                raise TypeError("All n values must be of type int")
            if order_parameters[1] < 0.0:
                raise ValueError("All n values must be non-negative ints")
            val_dict[f"K{order}"] = units.UnitFloat(order_parameters[0], units.ENERGY)
            val_dict[f"n{order}"] = order_parameters[1]
            val_dict[f"d{order}"] = units.UnitFloat(order_parameters[2], units.ANGLE)

        super().__init__(val_dict)


class LennardJones(InteractionFunction):
    r"""
    Dispersive Lennard-Jones interaction with the form:

    .. math::

        E = 4{\epsilon}[(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^6)]
        \qquad r < r_c

    Parameters
    ----------
    epsilon : float
        The LJ epsilon value in units of ``kJ mol^-1``
    sigma : float
        The LJ sigma value in units of ``Ang``
    **settings
        cutoff : float
            The distance in ``Ang`` at which the potential is cutoff
        long_range_solver : str
            The long range solver, either ``'PPPM'``, ``'PME'``, or ``'E'`` for
            Particle-Particle Particle-Mesh, Particle Mesh Ewald, or Ewald
            solvers

    Examples
    --------
    The following creates a ``Dispersion`` interaction with a ``LennardJones``
    functional form:

    .. highlight:: python
    .. code-block:: python

        lj = LennardJones(0.6502, 3.166)
        O_disp = Disperion(universe, (O.atom_type, O.atom_type), function=lj)
    """

    @inter_func_decorator(("epsilon", units.ENERGY), ("sigma", units.LENGTH))
    def __init__(self, epsilon: float, sigma: float, **settings: Any):

        super().__init__({"epsilon": epsilon, "sigma": sigma})
        self.cutoff = settings.get("cutoff")
        self.solver = settings.get("long_range_solver")


class NonBonded(InteractionFunction):
    @inter_func_decorator(
        ("charge", units.CHARGE),
        ("epsilon", units.ENERGY),
        ("sigma", units.LENGTH),
    )
    def __init__(self, charge: float, epsilon: float, sigma: float, **settings: Any):
        super().__init__({"charge": charge, "epsilon": epsilon, "sigma": sigma})

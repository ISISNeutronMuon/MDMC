.. _dev_doc_units-label:

Using units
===========
All physical quantities within MDMC should possess units.

Everything related to units in MDMC is defined in :py:mod:`MDMC.common.units`.
This includes:

- the system of units (in the :code:`SYSTEM` dictionary)
- A :class:`~MDMC.common.units.Unit` class, which derives from :code:`str` to
  add additional arithmetic methods (:code:`__mul__`, :code:`__div__`, and
  :code:`__pow__`) so that strings representing units (e.g. :code:`'Ang'`,
  :code:`'s'`) can be combined by these operations:
  :code:`Unit('Ang') * Unit('s') == Unit('Ang s')`
- :class:`~MDMC.common.units.UnitFloat` and
  :class:`~MDMC.common.units.UnitNDArray` classes, which derive from
  :code:`float` and :code:`numpy.ndArray` to add additional a
  :class:`~MDMC.common.units.Unit` object to the representation, by means of a
  :code:`unit` attribute.

So all physical quantities must be a :class:`~MDMC.common.units.UnitFloat` or a
:class:`~MDMC.common.units.UnitNDArray`, and have the correct
:class:`~MDMC.common.units.Unit` object as an attribute. This can be achieved by
having the unit as a property, rather than an attribute, and using one of the
following decorators (in :py:mod:`MDMC.common.decorators`):

- :class:`~MDMC.common.decorators.unit_decorator`: This should typically be the
  decorator that is used as it sets the property to be either a
  :class:`~MDMC.common.units.UnitFloat` or
  :class:`~MDMC.common.units.UnitNDArray`. It should be added to the property
  setter:

.. code-block:: python

  @property
  def velocity(self):

      return self._velocity

  @velocity.setter
  @unit_decorator(unit=units.LENGTH / units.TIME)
  def velocity(self, velocity):

      self._velocity = velocity

- :class:`~MDMC.common.decorators.unit_decorator_getter`: This only be used
  either where the property has no setter method or where the getter method
  performs a calculation before setting the value (which may result in the
  :class:`~MDMC.common.units.UnitFloat`/:class:`~MDMC.common.units.UnitNDArray`
  being cast to a :code:`float`/:code:`ndArray`). It should only be used in
  these cases as it is more expensive than
  :class:`~MDMC.common.decorator.unit_decorator`, as it initialises a
  :class:`~MDMC.common.units.UnitFloat`/:class:`~MDMC.common.units.UnitNDArray`
  object every time the getter is called. It should be added to the property
  getter:

.. code-block:: python

  @property
  @unit_decorator_getter(unit=units.LENGTH)
  def dims(self):

      return self._dims

In both of the above examples the unit passed to the decorator is a constant
from the :code:`SYSTEM` of units defined in the :py:mod:`MDMC.common.units`;
however it would be equally valid to define a unit in this argument:

.. code-block:: python

  from MDMC.common.units import Unit
  @property
  @unit_decorator_getter(unit=Unit('nm'))
  def dims(self):

      return self._dims

It is then obviously important that all functions and methods that use these
properties have the correct factors to ensure the unit conversions are correct.

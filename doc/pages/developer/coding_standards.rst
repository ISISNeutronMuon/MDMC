.. _dev_doc_coding_standards-label:

Coding Standards
================
Follow standards described in
`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_, except where it differs
from standards already adopted in surrounding code.  Some of these differences
are described below, as well as selected reminders of PEP 8 standards.

Line length
-----------
The line length in MDMC (including documentation) is 80 characters.

snake_case
----------
snake_case is used for all variables names, except when the variable contains an
abbreviation. So a variable related 





to a molecular dynamics (MD) engine is
MD_engine, rather than md_engine. The same applies for physical symbols, e.g. Q
for momentum transfer and E for energy should both be upper case.

Mathematical Operands
---------------------
With regards mathematical operands, favour a space separating both sides of an
operand:

.. code-block:: python

  E = h_bar * 1e18 * np.pi * np.arange(n) / (n * dt)

instead of:

.. code-block:: python

  E = h_bar*1e18*np.pi*np.arange(n)/(n*dt)


If an expression runs over two lines, prefer brackets to backslash, and align
expressions in the same manner as for mathematical formula (i.e. new line should
start with operand):

.. code-block:: python

  E = (h_bar * 1e18 * np.pi
       * np.arange(n) / (n * dt))

instead of:

.. code-block:: python

  E = h_bar * 1e18 * np.pi\
      * np.arange(n) / (n * dt)

or:

.. code-block:: python

  E = (h_bar * 1e18 * np.pi *
       np.arange(n) / (n * dt))

Method ordering
---------------
The ordering of methods in classes is as follows:

1. :code:`__new__`
2. :code:`__init__`
3. Other magic methods (:code:`__`)
4. Properties
5. Public methods
6. Private methods

This order is because magic methods, properties and public
methods make up the interface of the class. Properties must precede public
methods so that Sphinx creates them directly after Attributes.

Imports
-------
Imports should be grouped in the following order:

1. stdlib
2. third party libraries
3. local modules

Each group should be sorted alphabetically and should be separated by a newline.

Type Hints
----------
MDMC has adopted gradual typing, which means that there is an intention to
include `type hints <https://docs.python.org/3/library/typing.html>`_ in the
MDMC codebase. It is encouraged that any new code includes type hints for
function/method parameters and return values, and class variables, although code
will not be rejected if it does not include type hints. 

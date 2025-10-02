===============================
 Parametrised File Simulations
===============================

Why use parametrised files?
===========================

MDMC serves as a light wrapper around an MD engine in order to use it
to run calculations without having to implement its own integration
algorithms or force-fields. These underlying MD engines have their own
input schema and in order to provide a unified interface, MDMC
provides the ``Simulation`` classes to allow users to run simple
calculations without having to care about the infrastructure.

However, it is often the case that a user might have experience with a
particular MD engine or otherwise already have input files for the MD
engines. In this case, a user may not want to have to reimplement
their force-fields through the MDMC ``Simulation`` / ``Interaction``
system.

This is particularly important for cases where potentials are more
complex or use interactions or features not covered by the internal
MDMC mechanisms for constructing potentials.

How to use parametrised files
=============================

The parametrised file system replaces the need to define a
``Universe`` and ``Interactions`` as it is assumed that these are
constructed through the ``FileSimulation`` which handles all aspects
of the computation.

In general ``FileSimulation`` objects take the details of the
simulation from the input files provided, over-riding certain
parameters where necessary for MDMC to perform its different
phases. Each ``FileSimulation`` handles these differently (as in other
MDMC engine wrappers) and techniques that work for one cannot be
assumed to work for others.

This is a result of the different the different MD engines exposing
different levels of accessible information through their wrappers.

.. warning::

   While ``FileSimulation`` types may still expose some methods and
   properties which coincide with the MDMC internal MD engine wrappers
   (``Simulation``, ``Universe``, etc.), it should be assumed that
   the file-engines are completely independent of them.

Parametrised files themselves resemble the input files of the
respective MD engine, with a couple of minor markup sections to define
parameters to optimise or configurational parameters which will be set
internally by the ``FileSimulation``. These are described in detail
below.


Value Parameters
----------------

In the parametrised file, the file contains all the structure of the
original file with parameters to be fit/substituted surrounded by
curly braces and labelled.

These parameters have the following form:

.. code-block::

   {name: initial-value}

.. note::

   Only floating point numbers are able to be fitted due to the
   constraints of the MDMC fitting engine.

Reused Parameters
-----------------

It is possible that if a parameter is to be used multiple times in the
same file (e.g. you have multiple atoms which take the same
force-field or the force-field is used in some other function), you
can use the following form:

.. code-block::

   {name}

However, these parameters must have an initial value defined somewhere
in the relevant input files.

Internal Paramaeters
--------------------

In certain cases, some ``FileSimulation`` s may need to have extra
parameters which are substituted in by the engine. These parameters
are generally denoted by having names prefixed with ``_``. These
parameters do not need (and in most cases should not have) initial
values as these will be handled directly by the ``FileSimulation``.

.. code-block::

   {_internal_name}

It is also possible to define your own internal parameters,
these parameters can be passed through to ``FileSimulation``
constructors, ``minimize`` and ``run`` to define the settings in the
file.

Internal parameters need not be ``float`` s, but as a call to ``str``
(with ``repr`` fallback) is used before substitution into the file,
the result of this must be a valid parameter in the input file.

Example
-------

.. hint::
   .. literalinclude:: argon.field
      :emphasize-lines: 11

   An example parametrised file for a DLPoly argon simulation.

   Note this file contains components demarcated with ``{ }``.
   This indicates a parameter name and initial value to be fitted in MDMC.

How are Parameters Handled?
===========================

When parameters are read from parametrised files, they are stored in
the parameter parser (``FileSimulation.parser``) by the keyword given
in the original parametrised file.

It is possible to retrieve these
parameters, e.g. for the purpose of adding/modifying constraints, as
either:

- the stored ``Parameters`` object ``FileSimulationparameters``
- a raw dictionary ``FileSimulation.param_dict``

Changing Parameters In Code
---------------------------

If you wish to manually set/override parameters, it is important to
understand the distinction between ``as_dict`` and ``as_parameters``.

- ``parameters`` returns the actual ``Parameters`` object
  associated with the parser and as such changes will be immediately
  reflected in the output parameters.

  This does not fix parameters unless explicitly told to do so.

  .. note::

     From the :class:`FileSimulation` this is accessible as the :meth:`parameters`
     property.

- ``param_dict`` returns a copy and is not attached to the parser, so
  changes to the dictionary will be ignored unless a call is made to
  ``FileSimulation.parser.update_param_dict`` passing the updated
  dictionary.

  .. note::

     ``update_param_dict`` cannot add or remove parameters from the
     parser.

     ``update_param_dict`` will also automatically fix the
     underlying parameter, removing it from any future fitting.

     This is by design for safety.

     Attempting to set ``param_dict`` directly may lead to unexpected
     consequences.

Modifying Constraints
---------------------

By default, MDMC defines constrains parameters to be :math:`\pm{}20\%`
of their initial value. This is followed in parametrised file
simulations.

Since the parsed parameters are directly accessible, however, it is
possible to programmatically define constraints:

.. code-block:: python

   my_file_sim = FileSimulationSubclass("my_file.txt")
   params = my_file_sim.parameters
   # Fix value between 0 & 1
   params['my_favourite_param'].constraints = (0., 1.)

Engines Supporting Parametrised Files
=====================================

LAMMPS
------

The basic LAMMPS parametrised file engine requires a script which sets
up the simulation.

This has the following requirements:

- The chosen ``atom_style`` must have a ``charge``.
- It must contain the line:

  .. code-block::

     atom_modify map array
- Every atom must either have a valid species name through
  ``labelmap`` or a ``type_map`` argument must be passed to
  the ``LAMMPSFileSimulation`` object

.. hint::
   .. literalinclude:: argon.lmp
      :emphasize-lines: 20

   An example parametrised file for a LAMMPS argon simulation.

To start a LAMMPS calculation, you can provide a path to a single script in the configuration.

.. code-block:: Python

   simulation = LAMMPSFileSimulation("argon.lmp",
                                     traj_step=30,
                                     time_step=10.18893/2,
                                     type_map={1: "Ar"})

The main script can be supplemented by sub-scripts if you have split
your calculation into components, e.g. through a ``read_data`` or
``include`` statement. These are passed as extra positional arguments.

.. code-block:: Python

   simulation = LAMMPSFileSimulation("argon.lmp",
                                     "argon_pot.lmp",
                                     "argon_struct.data",
                                     traj_step=30,
                                     time_step=10.18893/2,
                                     type_map={1: "Ar"})

.. note::

   All available parameters in all input files are subject to
   substitution.

   This can be a helpful way of making transferable/re-usable files by
   isolating MDMC parameters to particular files and using core scripts
   to define the system state and ``include`` the potentials.

DLPoly
------

The basic DLPoly parametrised file engine requires the three usual
input files (``control``, ``field``, ``config``).

.. code-block:: Python

   simulation = DLPolyFileSimulation(control="argon.control",
                                     config="argon.config",
                                     field="argon.field",
                                     time_step=10.18893/2,
                                     traj_step=30,
                                     numprocs=4)

Optionally extra scripts can be passed through as optional keywords:

- ``equil_control`` -- Control file to use during the equilibration phase.
- ``minim_control`` -- Control file to use during the minimization phase.

This is mostly useful for defining different thermostats and
other parameters or intergration methods for these different stages.


.. note:: Any ``control`` parameters which will be set directly by the
   ``FileSimulation`` e.g. ``equilibration_steps``, these will be
   over-ridden by the MDMC ``FileSimulation`` engine wrapper.

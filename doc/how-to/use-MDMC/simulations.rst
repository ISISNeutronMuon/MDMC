.. _simulation-label:

Simulations
===========

Creating a Universe
-------------------
An MDMC simulation requires a :class:`~MDMC.MD.simulation.Universe`, and one or
more :class:`~MDMC.MD.structures.Atom` objects, which can be combined into
one or more :class:`~MDMC.MD.structures.Molecule` objects.  These atoms
and molecules must be added to a :class:`~MDMC.MD.simulation.Universe`.

Atoms can have bonded (:class:`~MDMC.MD.structures.Bond`,
:class:`~MDMC.MD.structures.BondAngle`,
:class:`~MDMC.MD.structures.Dihedral`) interactions or non-bonded
(:class:`~MDMC.MD.structures.Dispersive`,
:class:`~MDMC.MD.structures.Coulombic`) interactions applied to them.
For a simulation to run, a functional form must be specified for each
interaction: for instance a :class:`~MDMC.MD.structures.Dispersive`
interaction could have a :class:`~MDMC.MD.interaction_functions.LennardJones` or
a :class:`~MDMC.MD.interaction_functions.Buckingham` interaction function
applied to it.

It is also possible to:

 - :py:meth:`~MDMC.MD.simulation.Simulation.fill` and
   :py:meth:`~MDMC.MD.simulation.Universe.solvate` a Universe, as
   well as `fill it using Packmol <./notebooks/filling-with-packmol.ipynb>`_

 - :py:meth:`~MDMC.readers.configurations.read` in configuration and topology
   (except :class:`~MDMC.MD.structures.Dispersive` interactions) from
   files (see `Reading atoms from configuration files <./notebooks/read-configurations.ipynb>`_
   tutorial)

 - :py:meth:`~MDMC.gui.view` the configuration and topology in a GUI, to check
   the setup is correct

 - Use :py:meth:`~MDMC.MD.simulation.Universe.add_force_field` to apply
   established force fields (e.g. OPLSAA) to the topology to set the interaction
   functions and parameters (see
   `Applying a FieldField <../tutorials/applying-a-forcefield.ipynb>`_)


The MDMC interface for these objects is explained in the following notebooks:

.. toctree::
   :maxdepth: 2

   notebooks/creating-atomic-configurations.ipynb
   notebooks/defining-molecule-interactions.ipynb
   notebooks/solvating-a-universe.ipynb
   notebooks/read-configurations.ipynb
   notebooks/molecular-visualization.ipynb
   notebooks/applying-a-forcefield.ipynb
   notebooks/filling-with-packmol.ipynb

Indeed, as mentioned in the :ref:`introduction-label`, MDMC can be used purely to aid
the setup of MD simulations (without performing a refinement) via these methods.

Creating a Simulation
---------------------
Once the :class:`~MDMC.MD.simulation.Universe` is set up, it should be passed to
a :class:`~MDMC.MD.simulation.Simulation` object.  This controls the properties
of the simulation, such as which engine will be used and what the thermodynamic
conditions are. It is then possible to
:py:meth:`~MDMC.MD.simulation.Simulation.minimize` and
:py:meth:`~MDMC.MD.simulation.Simulation.run` the
:class:`~MDMC.MD.simulation.Simulation`.  The results of the simulation can then
be accessed through :class:`~MDMC.MD.simulation.Simulation`, for example via the
property :py:attr:`~MDMC.MD.simulation.Simulation.trajectory`, or through the
creation of observables such as the dynamic structure factor :math:`S(Q, \omega)``

These are explained in the following notebooks:

.. toctree::
   :maxdepth: 1

   notebooks/running-a-simulation.ipynb
   notebooks/creating-an-observable.ipynb
.. _simulation-label:

Simulations
===========
An MDMC simulation requires a :class:`~MDMC.MD.simulation.Universe`, and one or
more :class:`~MDMC.MD.structural_units.Atom` objects, which can be combined into
one or more :class:`~MDMC.MD.structural_units.Molecule` objects.  These atoms
and molecules must be added to a :class:`~MDMC.MD.simulation.Universe`.

Atoms can have bonded (:class:`~MDMC.MD.structural_units.Bond`,
:class:`~MDMC.MD.structural_units.BondAngle`,
:class:`~MDMC.MD.structural_units.Dihedral`) interactions or non-bonded
(:class:`~MDMC.MD.structural_units.Dispersive`,
:class:`~MDMC.MD.structural_units.Coulombic`) interactions applied to them.
For a simulation to run, a functional form must be specified for each
interaction: for instance a :class:`~MDMC.MD.structural_units.Dispersive`
interaction could have a :class:`~MDMC.MD.interaction_functions.LennardJones` or
a :class:`~MDMC.MD.interaction_functions.Buckingham` interaction function
applied to it.

Once the :class:`~MDMC.MD.simulation.Universe` is setup, it should be passed to
a :class:`~MDMC.MD.simulation.Simulation` object.  This controls the properties
of the simulation, such as which engine will be used and what the thermodynamic
conditions are. It is then possible to
:py:meth:`~MDMC.MD.simulation.Simulation.minimize` and
:py:meth:`~MDMC.MD.simulation.Simulation.run` the
:class:`~MDMC.MD.simulation.Simulation`.  The results of the simulation can then
be accessed through :class:`~MDMC.MD.simulation.Simulation`, for example via the
property :py:attr:`~MDMC.MD.simulation.Simulation.trajectory`.

For in-depth examples of these please see the related tutorials, particularly
`Building a Universe <../tutorials/building-a-universe.ipynb>`_ and
`Running a Simulation <../tutorials/running-a-simulation.ipynb>`_.

As mentioned in the :ref:`introduction-label`, MDMC can be used purely to aid
the setup of MD simulations, without performing a refinement.  It is possible
to:

 - :py:meth:`~MDMC.MD.simulation.Simulation.fill` and
   :py:meth:`~MDMC.MD.simulation.Universe.solvate` a Universe

 - :py:meth:`~MDMC.readers.configurations.read` in configuration and topology
   (except :class:`~MDMC.MD.structural_units.Dispersive` interactions) from CIF
   files (see `Reading atoms from configuration files <../tutorials/read-configurations.ipynb>`_
   tutorial)

 - :py:meth:`~MDMC.gui.view` the configuration and topology in a GUI, to check
   the setup is correct

 - Use :py:meth:`~MDMC.MD.simulation.Universe.add_force_field` to apply
   established force fields (e.g. OPLSAA) to the topology to set the interaction
   functions and parameters (see
   `Applying a FieldField <../tutorials/applying-a-forcefield.ipynb>`_)


.. rubric:: Related Tutorials

| `Building a Universe <../tutorials/building-a-universe.ipynb>`_
| `Reading atoms from configuration files <../tutorials/read-configurations.ipynb>`_
| `Applying a ForceField <../tutorials/applying-a-forcefield.ipynb>`_
| `Solvating a Universe <../tutorials/solvating-a-universe.ipynb>`_
| `Running a Simulation <../tutorials/running-a-simulation.ipynb>`_

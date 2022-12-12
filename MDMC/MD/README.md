# MD
This package contains everything required to run a molecular dynamics simulation using an external engine.  This includes the modules for defining the simulation configuration and the topology, the subpackage for defining the force fields, and the subpackage in which the molecular dynamics engine facades are defined.  Converting an MD engine trajectory into an MDMC CompactTrajectory is also contained within this package, although the CompactTrajectory class is in the trajectory_analysis package.

Additional engines facades can be added by creating a module with the engine name within the engine_facades subpackage.  This module must contain a class with the same name which implements the methods defined in the MDEngine class.

Additional force fields can be added by creating a module with the force field name within the force_fields subpackage.  This module must contain a class with the same name which implements the methods defined in the ForceField class.  Some engine facades will require modification to support additional force fields.

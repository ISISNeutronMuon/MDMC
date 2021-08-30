# trajectory_analysis
This package contains the module for defining the molecular dynamics trajectory, and a subpackage containing the observables that are calculated from the trajectory.  

Additional observables can be added by creating a module with the observable name within the observables subpackage.  This module must contain a class with the same name which implements the methods defined in the Observable class.

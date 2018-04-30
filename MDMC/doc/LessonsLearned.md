# MDMC v0.2 (Pilot) Lessons Learned

## MMTK

### Forcefields
Forcefields in MMTK are implemented using a database which contains common atoms, molecules, groups etc. Items not found within the database must be added using the database module.
The issue arises from the fact that in the database only the atom type and the charge are defined, and the forcefield parameters are contained elsewhere.  For instance, for SPCE the magnitudes of the point charges are defined within the database object, whereas the LJ parameters are defined within the SPCEFF.py (in the Forcefields module).  *Changing this parameter would require modification of how SPCE is implemented in MMTK.*  While for the AMBER forcefields the potential parameters are not hard coded into the forcefield class, they are within a separate file for AMBER data within the Forcefields module.  
The documentation for the base class states that each forcefield must provide a parameter set object which contains all possible parameters for the forcefield - *this parameter set object is not designed to be modified.*

The LJ parameters for the LJ forcefield (as opposed to those required for other forcefields, e.g. SPCE) are defined within the database for some elements e.g. Ar.  This is because the LJ forcefield is only designed to be applied to simple liquids.

To enable on the fly modification of SPCE force field parameters (LJ, bond strengths and bond angles) another version of the SPCEFF.py must be created - this is then passed to the MMTK Universe object after it has been created.  These parameters can then be set through the forcefield defined on the universe:

universe.forcefield().dataset.Params = (...)

This is not an ideal solution to this problem but is suitable for the pilot version.

It is not currently apparent how to modify the values of the charges on the fly.

*This overall lack of consistency in how the force field parameters are defined in MMTK might preclude its inclusion in the final version of MDMC*


### MD Simulation Properties
The various MD simulation software will also include analysis tools for calculating different simulation properties e.g. RMS.  Depending on the amount of work it will require, it would be best if the majority of these tools were included within MDMC, so that the analysis that can be performed is independent on the MD engines which was used for the simulation.  The alternative would be to support using the analysis tools of each MD engine.  While this will be trivial in some instances (e.g. MMTK), it will be more challenging in others.  It will also require a greater amount of work to create an interface for each additional MD engine.


### Atom ID
Potentially beneficial for each atom to be uniquely identified by an atomID (positive integer).  This should be useful for some MD engines such as GROMACS which require an atom ID, and may be useful for searching to fix forcefield parameters. Almost certainly required for setting bonds as well.


### Box creation
The simulation box should possess a size, shape, and boundary conditions.  It might be useful if the size can be defined in two ways:
* Specified by the user (either upon initialization or later)
* Determined based on the extremes of the atoms that have been included in the box.
The latter of these two might require some careful thought, however it would be useful in the case of liquids if the user can just specify a number of atoms and a box shape and the box size is determined for them.


### Hierarchy of structural units
It appears that a reasonable approach to creating structural units (e.g. atoms, groups, molecules) is to use a factory pattern. This would then allow the creation of any of these objects to occur in the same manner through the simulation box class (i.e. box.add(atom(...)) or box.add(molecule(...))).  This will also help when atoms or groups are added to molecules, and will allow easy extension to other structural units that have not currently been implemented.


### Charges
In many respects it makes sense to have the charge of each atom as an attribute of the atom object, as it is a parameter that each atom possesses and is unrelated to  any other object (at least generally - there may be special cases where this is not true).  This is the method that MMTK uses for defining the charge.  However, as the charge is a forcefield parameter, it is important that it is accessible in the same way as the bonded interaction parameters, so that they can be adjusted in a consistent manner.  Therefore there are two choices:
* Store the charge independently from the atom class, in a non-bonded force field class - has the drawback that a probable operation on an atom (accessing the charge) is no longer trivial.
* Having the charge as an attribute of the atom class, which can be modified from a non-bonded force field class - increases coupling between classes.
A further important consideration is whether to have a different force field object for each non-bonded interaction for each atom, even though in theory the charges of all identical atoms should be the same.  Having an object per atom would allow the charge on each atom to be varied individually, which might be useful in some situations - *discuss this at June 2018 developer meeting.*

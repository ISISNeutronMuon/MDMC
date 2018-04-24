# MDMC v0.2 (Pilot) Lessons Learned

## MMTK

###Forcefields
Forcefields in MMTK are implemented using a database which contains common atoms, molecules, groups etc. Items not found within the database must be added using the database module.
The issue arises from the fact that in the database only the atom type and the charge are defined, and the forcefield parameters are contained elsewhere.  For instance, for SPCE the magnitudes of the point charges are defined within the database object, whereas the LJ parameters are defined within the SPCEFF.py (in the Forcefields module).  *Changing this parameter would require modification of how SPCE is implemented in MMTK.*  While for the AMBER forcefields the potential parameters are not hard coded into the forcefield class, they are within a separate file for AMBER data within the Forcefields module.  
The documentation for the base class states that each forcefield must provide a parameter set object which contains all possible parameters for the forcefield - *this parameter set object is not designed to be modified.*

The LJ parameters for the LJ forcefield (as opposed to those required for other forcefields, e.g. SPCE) are defined within the database for some elements e.g. Ar.  This is because the LJ forcefield is only designed to be applied to simple liquids.

To enable on the fly modification of SPCE force field parameters (LJ, bond strengths and bond angles) another version of the SPCEFF.py must be created - this is then passed to the MMTK Universe object after it has been created.  These parameters can then be set through the forcefield defined on the universe:

universe.forcefield().dataset.Params = (...)

This is not an ideal solution to this problem but is suitable for the pilot version.

It is not currently apparent how to modify the values of the charges on the fly.

*This overall lack of consistency in how the force field parameters are defined in MMTK might preclude its inclusion in the final version of MDMC*

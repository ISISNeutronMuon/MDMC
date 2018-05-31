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
The latter of these two might require some careful thought, however it would be useful in the case of liquids if the user can just specify a number of atoms, density, and a box shape and the box size is determined for them.


### Hierarchy of structural units
It appears that a reasonable approach to creating structural units (e.g. atoms, groups, molecules) is to use a factory pattern. This would then allow the creation of any of these objects to occur in the same manner through the simulation box class (i.e. box.add(atom(...)) or box.add(molecule(...))).  This will also help when atoms or groups are added to molecules, and will allow easy extension to other structural units that have not currently been implemented.


### Charges
In many respects it makes sense to have the charge of each atom as an attribute of the atom object, as it is a parameter that each atom possesses and is unrelated to  any other object (at least generally - there may be special cases where this is not true).  This is the method that MMTK uses for defining the charge.  However, as the charge is a forcefield parameter, it is important that it is accessible in the same way as the bonded interaction parameters, so that they can be adjusted in a consistent manner.  Therefore there are two choices:
* Store the charge independently from the atom class, in a non-bonded force field class - has the drawback that a probable operation on an atom (accessing the charge) is no longer trivial.
* Having the charge as an attribute of the atom class, which can be modified from a non-bonded force field class - increases coupling between classes.

A further important consideration is whether to have a different force field object for each non-bonded interaction for each atom, even though in theory the charges of all identical atoms should be the same.  Having an object per atom would allow the charge on each atom to be varied individually, which might be useful in some situations. *discuss this at June 2018 developer meeting.*

A similar question arises for dispersive interaction parameters (e.g. LJ sigma), although for this it might be the case that the MD engine does not assign this for every atom, and instead just has a single set of dispersive parameters.  This is definitely the situation for MMTK.

### Python Version
Currently making code backwards compatible with Python 2.7 (with the exception of using enum module of stdlib and implementation of abstract classes), but this might lead to some restrictions - do we want this to be the case with the release version of MDMC? *discuss this at June 2018 developer meeting.*


### MD Engine Ensemble
Decided upon different simulator classes for different ensembles.  This will make it clear which ensemble is selected in each instance, whereas I think just passing a thermo/barostat to a 'simulate' class is less clear.


### Bond generation from pdb files
If MDMC is to read pdb files and generate topology from them, we will need to select a method for assigning bonds.  A common simple approach is to determine a distance cutoff below which bonds are assigned.  This can further be developed by having bond configurations for (and between) known structures (specifically residues).  Are there any other more sophisticated/general methods that can be applied?


### Format for storing topologies
As with MD engines, it would make sense for MDMC to store common topologies (and configurations?) internally, so that users are not required to define them.  This also enables users to add to the database of topologies, which could ideally be fed back into release versions in a simple manner.  We need to consider what the best format for storing these topologies is, particularly given the requirement that they need to be easily created by users. *discuss this at June 2018 developer meeting*


### Universe building approaches
There are two approaches for filling a universe with n copies of a structural unit:
* Creating the structural unit, adding it to the universe, adding a forcefield, and then copying the structural unit.  Just adding the forcefield to a single structural unit and then copying this will eliminate the need to assign an interaction function (and parameters) to every interaction, which will be quicker than the alternative:
* Creating the structural unit and then adding n copies of it to the universe.  Following this a forcefield would need to be applied to the universe, which would result in interaction functions being assigned to every interaction.
The first of these is preferential, however the second is potentially more natural from a user perspective.  *Therefore ideally the second of these (from the user perspective) should be setup to perform the first of these in the background i.e. the user should have to specify a forcefield when they perform the copy.*


### Determine cost of weakrefs
Currently the universe-atom and atom-interaction relationships contain cyclical references using weakrefs.


### MMTK LJ Parameters
MMTK returns three parameters for LJ interactions, where commonly there would be two.  The third parameter is undefined, and is usually hard coded as 0.


### Vector parameter passing
Use `*`args to allow vectors to be passed as both a list and individual floats.


### Universe bounding box position
MMTK defines it's universes centered around [0,0,0], rather than having this as a corner. Need to map positions when converting between MDMC and MMTK universe. Do we want our universes to be defined around or from [0,0,0]?  *discuss this at June 2018 developer meeting*


### Configurations, Trajectories and Histograms
Configuration currently stores both positions and velocities - do we want to allow it to store just one or the other for analysis i.e. of just position or just velocity observables.  Similarly do we want both to be stored in Trajectories?  GROMACS has different file formats which either come with or without velocity data.


### Named arguments
When creating examples/tutorials, use named arguments for all function/class arguments, as this is more explicit. 

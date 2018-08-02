# MDMC v0.2 (Pilot) Lessons Learned

# After Meeting 2 (Copenhagen)

### Refinement against total, coherent and incoherent dynamics structure factors
Typically neutron data will be compared against the calculated total DSF, as this is what is being measured; however, if it is known that either the coherent or incoherent terms will dominate then it might be preferable to refine the MD against whichever term dominates, as calculation of this term might be significantly quicker than calculating the total DSF.  For example, for water (calculated with nMOLDYN) the contribution of the DCSF to the total is <0.1 % for almost all Q,w, and only rises to 0.8% at low Q.


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


### Analysis tools (MD Simulation Properties)
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
In many respects it makes sense to have the charge of each atom as an attribute of the atom object, as it is a parameter that each atom possesses and is unrelated to any other object (at least generally - there may be special cases where this is not true).  This is the method that MMTK uses for defining the charge.  However, as the charge is a forcefield parameter, it is important that it is accessible in the same way as the bonded interaction parameters, so that they can be adjusted in a consistent manner.  Therefore there are two choices:
* Store the charge independently from the atom class, in a non-bonded force field class - has the drawback that a probable operation on an atom (accessing the charge) is no longer trivial.
* Having the charge as an attribute of the atom class, which can be modified from a non-bonded force field class - increases coupling between classes.

A further important consideration is whether to have a different force field object for each non-bonded interaction for each atom, even though in theory the charges of all identical atoms should be the same.  Having an object per atom would allow the charge on each atom to be varied individually, which might be useful in some situations - to clarify, currently force fields assume that atoms are identical if their environment is the same to first order, i.e. if they have the same bonds; however in reality the higher order environment may have a significant enough effect to be worth considering.  *discuss this at June 2018 developer meeting.*

A similar question arises for dispersive interaction parameters (e.g. LJ sigma), although for this it might be the case that the MD engine does not assign this for every atom, and instead just has a single set of dispersive parameters.  This is definitely the situation for MMTK.

### Python Version
Currently making code backwards compatible with Python 2.7 (with the exception of using enum module of stdlib and implementation of abstract classes), but this might lead to some restrictions - do we want this to be the case with the release version of MDMC? *discuss this at June 2018 developer meeting.*

Actually changed this so that it currently only uses 2.7 features, as MMTK uses 2.7


### MD Engine Ensemble
Decided upon different simulator classes for different ensembles.  This will make it clear which ensemble is selected in each instance, whereas I think just passing a thermo/barostat to a 'simulate' class is less clear.


### Bond generation from pdb files
If MDMC is to read pdb files and generate topology from them, we will need to select a method for assigning bonds.  A common simple approach is to determine a distance cutoff below which bonds are assigned.  This can further be developed by having bond configurations for (and between) known structures (specifically residues).  Are there any other more sophisticated/general methods that can be applied? *discuss this at June 2018 developer meeting*


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

A further consideration is the container type for storing data within Configuration, Trajectory and Histogram classes.  Currently this is a NumPy ordered array, which has the benefit that an object name and type are associated with each type of data i.e. all of the times in a trajectory can be recovered from the data variable with data['time'].  This has the advantage of being more explicit, however it could lead to potential drawbacks when plotting data (e.g. having to convert to a regular array).  *discuss this at June 2018 developer meeting*

Also, do we want histogram data to be stored with bin edges or bin centers?  NumPy outputs an array of values and an array of bin edges (i.e. len(values) + 1).


### Named arguments
When creating examples/tutorials, use named arguments for all function/class arguments, as this is more explicit.


### Trajectory filtering - frame vs time
In nMoldyn one of the parameters for both RDF and SQw is the frame (start, stop, step).  This is then translated into the time for calculations, and it is the time that is output to the user.  It feels more intuitive to me for the user to specify the time, rather than the number of frames - do we want to do this instead?  We could also output the total number of frames used, if this information is deemed relevant.  *discuss this at June 2018 developer meeting*


### Histogram generator
There isn't an obvious histogram algorithm that takes a generator as input, although this might be useful for us in the case of very large configurations - however the fastest histogram algorithms are in c, which would prohibit this.  Another possible solution for very large configurations would be forming several histograms from subsets of the configuration, and then summing these histograms.


### Histogram and Trajectory interactions
The interactions between the Histogram and Trajectory classes needs to be rethought, particularly with respect to time rebinning.  Currently rebinning a trajectory results in an array of trajectories being created, whereas it should essentially rebin in place.


### Calculating distances
Currently calculates distances between all atoms, which may be unnecessary depending on the size of the simulation and the required bound (either rmax/qmin).  Sorting atoms into boxes depending on this bound and then only calculating distances for atoms in the same box or adjacent boxes will eliminate unnecessary calculations.


### Experimental Observables
As with MD engines, experimental observables will use the factory pattern and introspection to allow for easy addition of new observables.  Currently the observable classes will contain the methods required to both calculate them from MD trajectories and read them in from experimental data.  Both of these aspects could be separated out into their own classes, but considering they would be strongly coupled to the ExperimentalObservable class, I am not sure about the benefits.


### Structure of experimental observables modules
nMOLDYN groups experimental observables together by type, e.g. structure, dynamics, scattering etc.  This would potentially be a good approach although in this case it might overcomplicate things with the introspection being used - it would particularly limit the ability for users to extend the package with their own experimental observables, although I am not sure what the probability of this is. *discuss this at June 2018 developer meeting*


### Uncertainties
An important consideration is how to deal with uncertainties (errors) in data, particularly given the need to perform operations on this when calculating the figure of merit.   While NumPy doesn't have a inbuilt mechanism for dealing with uncertainties, there is the appropriately named uncertainties package which extends NumPy to include arrays.  While in general I would prefer to only use major packages (such as NumPy or pandas) as they are unlikely to be discontinued, the uncertainties package was first released in 2008, so appears likely to be maintained.  Alternatively NumPy could be extended internally to MDMC to deal with uncertainties, to remove any dependencies. *discuss this at June 2018 developer meeting*


### Structure of readers module
As with the experimental observables module, readers could be grouped, for instance into the type of data that is being read e.g. configuration, topology, static or dynamic data.  This might be more beneficial in this instance, however the issue arises with formats that are used to store multiple different types of data (e.g. possibly hdf5?). *discuss this at June 2018 developer meeting*


### File output
Do we also want to be able to write to all of the file formats that we are supporting? *discuss this at June 2018 developer meeting*


### Units
Need to consider how best to deal with units - whether to have an inbuilt conversion (like MMTK) or just state which units will be used for each variable.


### Data in readers
The format in which data in readers is stored should be self descriptive, so that whatever calls a reader can extract the data in whatever manner it chooses with no further information.


### Uncertainty in MD trajectories
Currently the uncertainty in MD trajectories (or other data determined from MD simulations) is not considered - in the experimental observables the error is set to zero if the observable is calculated from MD. Is it possible to calculate uncertainty in MD trajectories? *discuss this at June 2018 developer meeting*


### MD Engine classes
Having looked back, potentially the MD engine classes (e.g. MMTK CubicUniverse) should instead just be a collection of functions which do the conversion between MDMC and the MD engine; this might help improve clarity as to what exactly they represent.  Obviously the abstract base class classes setup_universe, setup_simulation and run are excluded from this - they would be calling the functions.


### MMTK Trajectories
In MMTK, trajectories are stored with element information and a list of position vectors separately.  I would definitely prefer to avoid this.


### Isotropic systems
For isotropic systems, improvements in statistics can be achieved by averaging over orthogonal directions.  To allow this to be taken advantage of by non-expert users (while maintaining performance for expert users) I would suggest having an option/flag to specify if the system is isotropic.  If this is not set True/False then for the first calculation of FQt etc, it is calculated separately for each orthogonal direction and compared.  If they are within a certain tolerance, the system is assumed to be isotropic (could also repeat this test every n calculations of FQt etc).  This does have a small risk that the system would be incorrectly determined to be isotropic, therefore invalidating all refinements, however I think the probability is vanishingly small. *discuss this at June 2018 developer meeting*


### Figure of Merit Module
As we want MDMC to be able to fit multiple datasets, the refinement needs to use a figure of merit that reflects this.  Therefore there should be a multiple dataset figure of merit class, which contains n figures of merit which it uses to calculate the total figure of merit (with specific weighting for each dataset if required.)  This multiple figure of merit class should have the same interface as the individual ones, as it needs to expose the same methods to the minimizer.

Each (single) figure of merit also needs to check that it has one experimental dataset (flag in experimental observable) and one MD calculated dataset.  If in future other simulation methods (e.g. Monte Carlo, DFT) are used to get data against which MD data is compared, these would count as experimental datasets (however this flag is not set by the user, it is as a result of the dataset being read from a file).

Do we think it is likely that different FoMs will need to be used when fitting multiple datasets? *discuss this at June 2018 developer meeting*


### Minimizer
It is likely that all of the minimizers we use for MDMC will be variations of the Monte Carlo method, so rather than have an abstract Minimizer class from which others are derived, we could potentially have a single Minimizer class and just use different functions to determine whether or not a new state is accepted or rejected.  However, if we do determine another suitable class of minimizers, this might restrict our ability to extend the code without modifying it.  Therefore I think using an abstract base class tailored towards MD methods is the best choice, as then any divergence from MD methods can just override any required methods.


### Parameter constraints
There appear to be two obvious ways to denote which potential parameters are constrained:

* Within each interaction, denote whether each parameter is constrained or not.  Then the minimizer would need to iterate over all interactions and only change those parameters which are unconstrained.  The concern with this is that if the majority of parameters are constrained, then there is a lot of wasted iteration.  This could partially be mitigated by having references to only interaction that possess constrained parameters, however if there are a lot of interactions with only one unconstrained parameter, this will not result in a significant improvement.

* Have a separate parameter class which has a constrained attribute.  This will allow a reference to every constrained parameter to be passed to the minimizer.  The only concern would be that the list could become very large if there is a large number of unconstrained parameters, however I think it is unlikely that this situation would occur.

*discuss this at June 2018 developer meeting*


### MDMCControl
This class essentially just sets up the experimental observables from files before calling the minimizer - I don't really like it in it's current form.  I need to determine if there are other control aspects that need to be added to it, or if it shouldn't exist at all.

Another clunky element is that because the experimental observables are instantiated with MDMCControl, but the minimizer needs to be passed these, then either one of the following has to occur:

* Either an instance of the minimizer with no attributes needs to be passed to MDMCControl;

* Or a string establishing whichever minimizer is to be used needs to be passed.

The first is unappealing because the initialization of the minimizer is essentially wasted, and attributes then have to be set separately.  The second is unappealing because an MD engine instance is also passed to MDMCControl, and I think mixing up passing instances and what is essentially a keyword lacks clarity.

It also currently has a "while True" loop with break conditions.  In this instance I feel this is valid, however this should be discussed. *discuss this at June 2018 developer meeting*

A further question of software philosophy which relates to the above discussion concerns how settings should be passed to instances within MDMCControl.  For example, if the FigureOfMeritCalculator has some settings, there are two ways in which they could be set:

* Either the class in instantiated by the user and passed to MDMCControl;

* Or a collection of settings are passed to MDMC.  

*discuss this at June 2018 developer meeting*


### Distribution of parameter changes
MDMC v0.1 uses a uniform RNG to determine the change in each parameter - need to determine if this is the best approach for sampling parameter phase space.


### Flexibility of FoM calculations
The most flexible structure for calculating FoMs would allow each dataset to use a different FoM when using multiple datasets.  However allowing this additional flexibility will require additional code and may make the common user case more complicated to select - is it an option that is likely to be used? *discuss this at June 2018 developer meeting*


### Analysis
Minimizer (or MDMCControl) needs to output some metadata about MC e.g. ratio of accepted/rejected steps, correlation between parameters etc.


### Configuration storing
MDMCv0.1 stored the configuration from the previous 'good' parameter set (i.e. the last MC accepted parameter set), and the MD configuration is reverted back to this if the MC step is rejected.  However based on the assumption that the system is kept in equilibrium by the small parameter changes, I'm not sure how necessary this is.  Avoiding doing it would save a significant amount of memory for large systems, and might also be advantageous in terms of interacting with certain MD engines.

However they may also be advantages to keeping the configuration.  It would be best to support both and allow the user to choose (could also limit it based on simulation size). *discuss this at June 2018 developer meeting*


### Attitude to scripts
As the end goal is developing software with a GUI, how user friendly do scripted arguments need to be e.g. is it reasonable to expect the user to provide a predicate to a filter, or should there be simple options for common predicates?  Does the ability to script for MDMC need to be suitable for inexpert users?

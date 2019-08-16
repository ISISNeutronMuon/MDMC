# Solvating an MDMC Universe

For simulating systems in solution, MDMC's inbuilt ``solvate`` method can be used to add solvent molecules of a desired density to your universe.

# Create the Universe
----
See the [_Building a Universe_] (https://github.com/MDMCproject/MDMCv0.2_pilot/wiki/Building-a-Universe)  wiki for details on how to create a Universe of your own specifications. For the purposes of this tutorial, we will be solvating a simple universe that contains only 4 Hydrogen molecules.

```
# Import the Atom, Molecule, and Universe classes
# Import the HarmonicPotential class (needed later to create a Bond)
from MDMC.MD.simulation import Universe
from MDMC.MD.structural_units import Atom, Molecule
from MDMC.MD.interaction_functions import HarmonicPotential

# Initialise a Universe with dimensions in Ang
universe = Universe([10.0, 15.0, 20.0])

# Create a pair Hydrogen Atoms
H1 = Atom('H')
H2 = H1.copy(position=[1., 1., 1.])

# Initialise a H-H Bond
HH_bond = Bond(H1, H2, function=HarmonicPotential((1., 'Ang'), 
												  (100., 'kJ / mol Ang^2')))

# Make a H2 Molecule
H2_mol_1 = Molecule(atoms=[H1, H2], interactions=[HH_bond])

# Create 3 copies of the Molecule at different positions
H2_mol_2 = H2_mol_1.copy(position=[3.5, 3.5, 3.5])
H2_mol_3 = H2_mol_1.copy(position=[6.5, 6.5, 6.5])
H2_mol_4 = H2_mol_1.copy(position=[9.5, 9.5, 9.5])

# Add the 4 Hydrogen Molecules to the Universe
for molecule in [H2_mol_1, H2_mol_2, H2_mol_3, H2_mol_4]:
	universe.add_structural_unit(molecule)
```

# Solvating the Universe
----
MDMC's solvate method accepts 3 main parameters; ``density``, ``tolerance``, and ``solvent`` (as well as some ``settings``). An example call to solvate the Universe created above would therefore be:

```
universe.solvate(0.6, tolerance=1, solvent='SPCE')
```  

## Parameters 

Explanations of each parameter passed to ``solvate`` can be found below. Each is specified


### 1. ``density``
The desired density of the bulk solvent in your universe, in MDMC units of **amu / Ang ^ 3**. See this tutorial **.....LINK TO UNIT CONVERSION WIKI.....** for instructions on how to convert your density into MDMC units.


**Note**: with this parameter you are specifying the **bulk density of the solvent**. If you have any solute molecules already present in your universe, the total density of your universe after solvation will be higher than the desired density you pass to ``solvate`` (plus or minus the tolerance you pass, see below).

In the above example, passing the density as ``0.6`` means that the universe will be solvated with SPCE water with a bulk density of 0.6 amu / Ang ^ 3 (+/- the tolerance). The density of the universe **in total will be greater** than 0.6 amu / Ang ^ 3 by the contribution the mass of 4 Hydrogen molecules has to the density.



### 2. ``tolerance``
With this parameter you can specify the percentage tolerance of the ``density`` that you would like to be achieved for the bulk density of the solvent. 

For the above example, passing a density of ``0.6`` and setting ``tolerance=1`` will achieve a bulk solvent density of:

* (0.6 amu / Ang ^ 3)  +/-  1 %
* equivalent to (0.6  +/-  0.006) amu / Ang ^ 3

**Note**: the tolerance has a default value of 1 %. Setting the tolerance to anything lower than increases the risk of ``solvate`` not converging to within the tolerance of the desired bulk solvent density.



### 3. ``solvent``

In all cases, the following parameters must the desired density of the solvent you would like solvate with must be passed as a parameter in addition to a tolerance level within which you 

#### a) Using a solvent with pre-defined coordinates

MDMC has a few in-built solvents that can be used to solvate your Universe. These have pre-defined atomic coordinates, Bonds, BondAngles, NonBondedInteractions, and that have been generated in simulation.

Currently, the in-built solvents you can choose from are:  

* SPCE water

##### Example: solvating with SPCE water

An example of solvating with in-built SPCE water:

```
universe.solvate(0.6, tolerance=1, solvent='SPCE')
```

#### b) Specifying a StructuralUnit.
You can also create an Atom or Molecule StructuralUnit which with to solvate the universe. Ensure that the StructuralUnit you are specifying has the correct bonded and nonbonded interactions


### 4. ``**settings``

#### a) ``constraint_algorithm``

You can specify ConstraintAlgorithm which is applied to the Universe. If specifying ``solvent`` as a string representing one of the in-built solvents (i.e. 'SPCE'), then a ``Shake(1e-4, 100)`` constraint algorithm is automatically applied.

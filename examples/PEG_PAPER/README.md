# The description of files

For example `pgme_280K_30W_125_441.pdb`: here 280 K is the temperature of simulation (pressure was 1 atm). 30 W - 30 wt % of water. 
125 is the number of PGME molecules in the system. 441 is the number of water molecules.

Systems with doubled numbers of water molecules are meant to be for checking out the performance of MDMC on smaller and bigger systems.

PGME.itp is the file from Gromacs software which can be used for modeling larger systems with PGME as Gromacs is well-parallelized across different nodes and GPUs as well.

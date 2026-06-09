import copy
import os

from MDMC.MD import *
from MDMC.MD.interactions import NonBondedForce
from MDMC.control import Control
from MDMC.refinement.FoM.FoM_abs import ObservablePair
from MDMC.trajectory_analysis.observables.sqw import SQw

# Currently MDMC uses OMP_NUM_THREADS to control the number of processes
# in the sqw calculation
os.environ["OMP_NUM_THREADS"] = "4"

# Build universe
# Cubic universe of side:
# 18.6270199 A is 216 water molecules
# 21.731523217 is 343 water molecules
# 24.83602653 is 512 water molecules
universe = Universe.from_pdb_file("7rsa_nowater.pdb")

for atom_type in universe.unique_atom_types:
    NonBondedForce(
        universe,
        atom_type,
        cutoff=10.0,
        ewald=1e-6,
        function=NonBonded(charge=0.0, epsilon=1.0, sigma=3.0),
    )

print(universe)
print(universe.molecule_list)

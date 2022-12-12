from MDMC.MD import *

# Define the unique atoms using the ForceField atom_type
# These can be seen in the oplsaa.dat file (MDMC/MD/force_fields/data/oplsaa.dat)
# The H1 atom will be copied after the bond and bond angles have been defined
HC1 = Atom('H', position=[-0.7006,  0.3636,  0.8900], name='98', charge=0., atom_type=1)
C = Atom('C', position=[-0.3366, -0.1504,  0.0000], name='99', charge=0., atom_type=2)
O = Atom('O', position=[ 1.0849, -0.1713,  0.0000], name='96', charge=0., atom_type=3)
HO = Atom('H', position=[ 1.3606,  0.7699,  0.0000], name='97', charge=0., atom_type=4)
# Create the bonds with harmonic potentials
CH_bond = Bond(C, HC1)
CO_bond = Bond(C, O, constrained=True)
OH_bond = Bond(O, HO)

# Create the H-C-O and H-O-C bond angles
HCO_angle = BondAngle((HC1, C, O))
HOC_angle = BondAngle((HO, O, C))

# Create the H-C-O-H dihedral
HCOH_dihedral = DihedralAngle((HC1, C, O, HO))

# Duplicate the HC1 atom
HC2 = HC1.copy(position=[-0.7006,  0.3636, -0.8900])

# Create an HCH bond angle
HCH_angle = BondAngle((HC1, C, HC2))

# Duplicate the HC1 atom again
# This atom will have all bond (CH_bond) and bond angles (HCO_angle and HCH_angle) defined
HC3 = HC1.copy(position=[-0.7076, -1.1754,  0.0000])

# Create the methanol Molecule
methanol = Molecule(atoms=[HC1, HC2, HC3, C, O, HO])

# Create a universe and add the methanol
universe = Universe(dimensions=15.0, constraint_algorithm=Shake(1e-5, 100), electrostatic_solver=PPPM(accuracy=1e-4))
universe.fill(methanol, num_density=0.01)

# Add dispersion interactions (no dispersion for HO)
HC_disp = Dispersion(universe, (1, 1), cutoff = 8.0, vdw_tail_correction=True)
C_disp = Dispersion(universe, (2, 2), cutoff = 8.0, vdw_tail_correction=True)
O_disp = Dispersion(universe, (3, 3), cutoff = 8.0, vdw_tail_correction=True)
HO_disp = Dispersion(universe, (4, 4), cutoff = 8.0, vdw_tail_correction=True)
universe.add_force_field('OPLSAA')
# decrease all cutoffs of the nonbonded interactions
for interaction in universe.nonbonded_interactions:
    interaction.cutoff=8.0

# Create the simulation
simulation = Simulation(universe, engine='lammps', time_step=1., temperature=300.,
                        pressure=101325., traj_step=10, thermostat='nose',
                        barostat='nose', t_damp=100, p_damp=1000)

# Run a minimization and equilibration
simulation.minimize(n_steps=100000)
# Warning: do not run the equilibration for much more than 100 steps, because the above
# geometry has not been optimised and may lead to diverging simulations. To run a longer
# equilibration, a better starting geometry needs to be chosen.
simulation.run(n_steps=100, equilibration=True)

import parmed as pmd
import openmm as mm
from openmm import app, unit

# Input files
topology_file = "topol.top"
coordinates_file = "pgme_280K_50W_125_1028.pdb"

# Read the complete GROMACS topology and coordinates.
# The topology must include the correct force-field files.
structure = pmd.load_file(
    topology_file,
    xyz=coordinates_file
)

# Report topology counts.
print("Atoms:", len(structure.atoms))
print("Residues:", len(structure.residues))
print("Bonds:", len(structure.bonds))
print("Angles:", len(structure.angles))
print("Periodic dihedrals:", len(structure.dihedrals))
print("RB torsions:", len(structure.rb_torsions))

# Check for missing coordinates.
if any(atom.xx is None or atom.xy is None or atom.xz is None
       for atom in structure.atoms):
    raise ValueError("Some atoms are missing coordinates.")

# Check periodic box information.
print("Box:", structure.box)

if structure.box is None:
    raise ValueError(
        "No periodic box was found. Check the PDB CRYST1 record "
        "and the original GROMACS box information."
    )

# Construct the OpenMM system.
# IMPORTANT: adjust these settings to match your original npt.mdp.
system = structure.createSystem(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
    removeCMMotion=False
)

# Save the OpenMM System.
with open("system.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(system))

# Save the coordinates and topology in OpenMM-compatible formats.
with open("system.pdb", "w") as f:
    app.PDBFile.writeFile(
        structure.topology,
        structure.positions,
        f,
        keepIds=True
    )

print("\nOpenMM system saved to system.xml")
print("Particles:", system.getNumParticles())

print("\nOpenMM forces:")
for force in system.getForces():
    print(type(force).__name__)

    if isinstance(force, mm.RBTorsionForce):
        print("  RB torsions:", force.getNumTorsions())

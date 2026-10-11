import openmm as mm
from openmm import app, unit

pdb = app.PDBFile("pgme_280K_50W_125_1028.pdb")

with open("system.xml") as f:
    system = mm.XmlSerializer.deserialize(f.read())

integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
platform = mm.Platform.getPlatformByName("CPU")

simulation = app.Simulation(
    pdb.topology,
    system,
    integrator,
    platform
)

simulation.context.setPositions(pdb.positions)

state = simulation.context.getState(getEnergy=True)
energy = state.getPotentialEnergy()

print("OpenMM potential energy:", energy)

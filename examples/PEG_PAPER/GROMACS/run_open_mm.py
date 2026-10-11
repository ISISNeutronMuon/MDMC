import openmm as mm
from openmm import app, unit
import sys
import os

# --------------------------------------------------
# Input files
# --------------------------------------------------
PDB_FILE = "pgme_280K_50W_125_1028.pdb"
SYSTEM_XML = "system.xml"

# --------------------------------------------------
# Simulation settings
# --------------------------------------------------
TEMPERATURE = 280 * unit.kelvin
PRESSURE = 1.013 * unit.bar
TIMESTEP = 2.0 * unit.femtoseconds

N_STEPS = 200
REPORT_INTERVAL = 50
CHECKPOINT_INTERVAL = 50

# GROMACS tau-t = 0.5 ps is not an exact Langevin equivalent.
FRICTION = 2.0 / unit.picosecond

# --------------------------------------------------
# Read coordinates and converted OpenMM system
# --------------------------------------------------
pdb = app.PDBFile(PDB_FILE)

with open(SYSTEM_XML, "r") as f:
    system = mm.XmlSerializer.deserialize(f.read())

if system.getNumParticles() != pdb.topology.getNumAtoms():
    raise ValueError(
        f"Particle mismatch: system has {system.getNumParticles()}, "
        f"PDB has {pdb.topology.getNumAtoms()} atoms"
    )

print("Particles:", system.getNumParticles())

# --------------------------------------------------
# Check periodic box
# --------------------------------------------------
if pdb.topology.getPeriodicBoxVectors() is None:
    raise ValueError("PDB has no periodic box vectors.")

system.setDefaultPeriodicBoxVectors(
    *pdb.topology.getPeriodicBoxVectors()
)

# --------------------------------------------------
# Add an isotropic Monte Carlo barostat
# --------------------------------------------------
# Do not add a second barostat if one is already present.
existing_barostats = [
    force for force in system.getForces()
    if isinstance(force, mm.MonteCarloBarostat)
]

if existing_barostats:
    print("MonteCarloBarostat already present.")
else:
    system.addForce(
        mm.MonteCarloBarostat(
            PRESSURE,
            TEMPERATURE,
            25
        )
    )

# --------------------------------------------------
# Integrator
# --------------------------------------------------
integrator = mm.LangevinMiddleIntegrator(
    TEMPERATURE,
    FRICTION,
    TIMESTEP
)

# --------------------------------------------------
# Select platform
# --------------------------------------------------
platform_name = "CUDA"

try:
    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {"Precision": "mixed"}
except Exception:
    platform = mm.Platform.getPlatformByName("CPU")
    properties = {}

print("Platform:", platform.getName())

# --------------------------------------------------
# Create simulation
# --------------------------------------------------
simulation = app.Simulation(
    pdb.topology,
    system,
    integrator,
    platform,
    properties
)

simulation.context.setPositions(pdb.positions)

# This is appropriate only when no initial velocities are available
# and you intentionally want to generate velocities at 280 K.
simulation.context.setVelocitiesToTemperature(TEMPERATURE)

# --------------------------------------------------
# Output files
# --------------------------------------------------
simulation.reporters.append(
    app.DCDReporter(
        "trajectory.dcd",
        REPORT_INTERVAL
    )
)

simulation.reporters.append(
    app.StateDataReporter(
        "energy.log",
        REPORT_INTERVAL,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        speed=True,
        separator="\t"
    )
)

simulation.reporters.append(
    app.CheckpointReporter(
        "checkpoint.chk",
        CHECKPOINT_INTERVAL
    )
)

# --------------------------------------------------
# Initial energy
# --------------------------------------------------
state = simulation.context.getState(
    getEnergy=True,
    getPositions=True
)

print(
    "Initial potential energy:",
    state.getPotentialEnergy()
)

# --------------------------------------------------
# Run simulation
# --------------------------------------------------
print(f"Running {N_STEPS} steps ({N_STEPS * 0.002:.1f} ps)...")

simulation.step(N_STEPS)

# --------------------------------------------------
# Save final coordinates
# --------------------------------------------------
state = simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True
)

with open("final.pdb", "w") as f:
    app.PDBFile.writeFile(
        simulation.topology,
        state.getPositions(),
        f,
        keepIds=True
    )

simulation.saveState("final_state.xml")

print("Simulation completed.")
print("Final potential energy:", state.getPotentialEnergy())
print("Files written:")
print("  trajectory.dcd")
print("  energy.log")
print("  checkpoint.chk")
print("  final.pdb")
print("  final_state.xml")

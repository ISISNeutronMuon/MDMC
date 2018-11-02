"""Script for running debug"""

import numpy as np
from scipy.interpolate import interp2d

import MDMC.MD.simulation as sim
import MDMC.MD.structural_units as su
import MDMC.MD.force_fields as ff
import MDMC.refinement.minimizer as minim
from MDMC.control.control import MDMCControl

from tests.test_data import data

# Build universe
# Cubic universe of side 9.32 A is 27 water molecules, 24.86 is 512 water molecules
side = 9.32
universe = sim.Universe(dimensions=(side, side, side),
                        shape=sim.Shape.orthorhombic)
H1 = su.Atom('H', mass=1.008)
H2 = su.Atom('H', position=(1.51390, 0., 0.), mass=1.008)
O = su.Atom('O', position=(0.75695, 0., 0.58588), mass=16.000)
water_mol = su.Molecule(position=(0, 0, 0),
                        velocity=(0, 0, 0),
                        atoms=[H1, H2, O],
                        interactions=[su.Bond(H1, O),
                                      su.Bond(H2, O),
                                      su.Dispersion(O),
                                      su.BondAngle(atoms=[H1, O, H2])],
                        name='water')
# Has a smaller number density
universe.fill(water_mol, force_field="SPCE", num_density=0.0335)

# Randomly change parameters by +/- value * fac
# fac = 0.2
# for p in universe.parameters:
#     if p.interactions_name != 'Coulombic' and p.interactions_name != 'BondAngle':
#         p.value = p.value + p.value * (np.random.random() * (fac * 2) - fac)
#     else:
#         p.fixed = True

# MD Engine setup
md_engine = sim.NVESimulation(universe,
                              engine="mmtk",
                              time_step=1.0,
                              temperature=263.,
                              integrator='velocity_verlet',
                              lj_options=1.2,
                              es_options='ewald',
                              minimizer='steepest_descent',
                              traj_step=500)

# Energy Minimization and equilibration
md_engine.minimize(n_steps=5000)
print "Minimization Complete"

md_engine.run(n_steps=5000, equilibration=True)
print "Equilibration Complete"

# Setup refinement

# exp_datasets is a list of dictionaries with one dictionary per experimental dataset
exp_datasets = [{'file_name':'/Users/thomasfarmer/Library/virtualenv/virtualenvMMTK/bin/nMOLDYN-3.0.8/RunFiles/water/spce/for_MDMC/testing_MDMC_output/NVE/DISF_water27_15000steps_spce.nc',
                 'type':'SQw',
                 'reader':'netCDF',
                 'weight':1.}]

# Fit parameters is a set of all unique fit parameters in the universe which can then be filtered.
fit_params = set([p for p in universe.parameters if p.fixed is False])
control = MDMCControl(MD_engine=md_engine,
                      exp_datasets=exp_datasets,
                      fit_params=fit_params,
                      MC_norm=1,
                      minimizer_type="MMC",
                      MD_steps=15000,
                      t_resolution=114.,
                      cell=md_engine.universe.dims)

# Run refinement
control.refine(n_steps=0)

from netCDF4 import Dataset
from MDMC.tests.test_data import data
import numpy as np

file = Dataset(data.OBS_DATA['SQw_coh'],'r')
Q_ref = np.array(file.variables['q'][:])
w_ref = np.array(file.variables['angular_frequency'][:])
t_ref = np.array(file.variables['time'][:])
FQt_ref = np.array(file.variables['Fqt-total'][:])
SQw_ref = np.array(file.variables['Sqw-total'][:])

from MMTK.Trajectory import Trajectory
import MDMC.trajectory_analysis.observables.obs_factory as of
import MDMC.MD.engine_facades.mmtk as mmtk
import numpy as np
import nMOLDYN.Mathematics.ReciprocalSpace as RS

trajectory = Trajectory(None, "/Users/thomasfarmer/Library/virtualenv/virtualenvMMTK/bin/MMTK-2.7.10/Simulations/Water/spce/263K/water343_108929steps_spce.nc")
MDMC_traj = mmtk.convert_trajectory(trajectory, slice=slice(0, 723, 7)) # (0, 1547, 15) for 1.5 fs step size
SQw = of.ObservableFactory.create_observable('SQw')
cell = trajectory.universe.cellParameters() * 10.
n_Q = 10
Q_values = np.arange(0.3, 3.6, 0.1)
SQw.calculate_from_MD(MDMC_traj, Q_values = Q_values, cell = cell, t_resolution = 141.)

from netCDF4 import Dataset

file = Dataset('/Users/thomasfarmer/Library/virtualenv/virtualenvMMTK/bin/nMOLDYN-3.0.8/RunFiles/water/spce/263K/DISF_water343_108929steps_spce.nc', 'r')
Q_ref = np.array(file.variables['q'][:])
w_ref = np.array(file.variables['angular_frequency'][:])
t_ref = np.array(file.variables['time'][:])
FQt_ref = np.array(file.variables['Fqt-total'][:])
SQw_ref = np.array(file.variables['Sqw-total'][:])

import numpy as np
from scipy.interpolate import interp2d

import MDMC.MD.simulation as sim
import MDMC.MD.structural_units as su
import MDMC.MD.force_fields as ff
import MDMC.refinement.minimizer as minim
from MDMC.control.control import MDMCControl

from tests.test_data import data

# Build universe
universe = sim.Universe(dimensions=(9.32, 9.32, 9.32), shape=sim.Shape.orthorhombic)
H1 = su.Atom('H', mass=1.008)
H2 = su.Atom('H', position=(1.51390, 0., 0.), mass=1.008)
O = su.Atom('O', position=(0.75695, 0., 0.58588), mass=16.000)
water_mol = su.Molecule(position=(0, 0, 0),
                        velocity=(0, 0, 0),
                        atoms=[H1, H2, O],
                        interactions=[su.Bond(H1, O),
                                      su.Bond(H2, O),
                                      su.Dispersion(O),
                                      su.BondAngle(atoms=[H1, O, H2])],
                        name='water')
universe.fill(water_mol, force_field="SPCE", num_density=0.0333679)

# MD Engine setup
md_engine = sim.NVESimulation(universe,
                              engine="mmtk",
                              time_step=1,
                              temperature=263.,
                              integrator='velocity_verlet',
                              lj_options=1.2,
                              es_options={'method':'ewald'},
                              minimizer='steepest_descent')

# Energy Minimization and equilibration
md_engine.minimize(n_steps=10000)
md_engine.run(n_steps=200)

# Setup refinement

# exp_datasets is a list of dictionaries with one dictionary per experimental dataset
exp_datasets = [{'file_name':data.READER_DATA['LAMPSQw'],
                 'type':'SQw',
                 'reader':'LAMPSQw',
                 'weight':1.}]

# Fit parameters is a set(?) of all unique fit parameters in the universe which can then be filtered.
fit_params = universe.parameters
control = MDMCControl(MD_engine=md_engine,
                      exp_datasets=exp_datasets,
                      fit_params=fit_params,
                      MC_norm=1,
                      minimizer_type="MMC",
                      MD_steps=103,
                      t_resolution=30.)

# Bertil Halle water data is non-symmetric, and has a non-rectangular grid with
# a non-uniform E step.
# To account for this, a limited E is used and undefined errors are set to zero
# for the purposes of interpolation.
# This should really be performed before the data is read into control - the
# final step is a reflection of this as the MD observable is changed to match
# the new independent variables of the experimental observable
exp_obs = control.observable_pairs[0].exp_obs
Q = exp_obs.Q
E_range = (exp_obs.E >=0)
E = exp_obs.E[E_range]
SQw = np.array([Sw[E_range] for Sw in exp_obs.SQw])
SQw_err = np.array([Sw_err[E_range] for Sw_err
                          in exp_obs.SQw_err])
SQw_fun = interp2d(E, Q, SQw)
SQw_err_zero = SQw_err
SQw_err_zero[SQw_err == np.float('inf')] = 0
SQw_err_fun = interp2d(E, Q, SQw_err_zero)
# Use the largest step size from the E data for the uniform step size
E_step = max([E[i] - E[i-1] for i in np.arange(len(E) - 1) + 1])
E_uniform = np.arange(E[0], E[-1], E_step)
SQw_uniform = SQw_fun(E_uniform, Q)
SQw_err_uniform = SQw_err_fun(E_uniform, Q)
SQw_err_uniform[SQw_err_uniform == 0.] = np.float('inf')
control.observable_pairs[0].exp_obs.independent_variables = {'E':E_uniform,
                                                             'Q':Q}
control.observable_pairs[0].exp_obs._dependent_variables = {'SQw':SQw_uniform}
control.observable_pairs[0].exp_obs._errors = {'SQw':SQw_err_uniform}
control.observable_pairs[0].MD_obs.independent_variables = {'E':E_uniform,
                                                            'Q':Q}

# Run refinement
control.refine(n_steps=0)







import MDMC.MD.interaction_functions as ifu
import MDMC.MD.structural_units as su
import MDMC.MD.simulation as sim
import MDMC.MD.force_fields as ff
import MDMC.MD.engine_facades.mmtk as mmtkf
import MDMC.trajectory_analysis.trajectory as tr
from MDMC.readers.LAMPSQw import LAMPSQw
import MDMC.trajectory_analysis.observables.obs_factory as eof
from tests.test_data import data
from MDMC.utilities import plot

from timeit import timeit
from copy import deepcopy

from MMTK import *
from MMTK.Minimization import SteepestDescentMinimizer

UNIVERSE_DIMS = (10.,10.,10.)
UNIVERSE_SHAPE = sim.Shape.orthorhombic
WATER_POSITION = (1.,2.,3.)
WATER_VELOCITY  = (3.,2.,1.)
WATER_NUM_DENSITY = 0.0333679

TIME_STEP = 1
TEMPERATURE = 300
N_STEPS = 10
INTEGRATOR = 'velocity_verlet'
LJ_OPTIONS = 1.2
ES_OPTIONS = {'method':'ewald'}
MINIM = 'steepest_descent'
MINIM_STEP_SIZE = 0.05
MINIM_STEPS = 10

universe = sim.Universe(UNIVERSE_DIMS,UNIVERSE_SHAPE)
H1 = su.Atom('H',mass=1.008)
H2 = su.Atom('H',position=(0.151390,0.,0.), mass=1.008)
O = su.Atom('O',position=(0.075695,0.,0.058588),mass=16.000)
water_molecule = su.Molecule(position=WATER_POSITION,velocity=WATER_VELOCITY,atoms=[H1,H2,O],
                    interactions=[su.Bond(H1,O),su.Bond(H2,O),
                                    su.Dispersion(O)],name='water')
water_molecule.add_interaction(su.BondAngle(atoms=[H1,O,H2]))

# universe.add_structural_unit(water_molecule)
# universe.add_force_field("SPCE")

universe.fill(water_molecule,force_field="SPCE",num_density=WATER_NUM_DENSITY)

NVESim = sim.NVESimulation(universe,'mmtk',time_step = TIME_STEP,
    temperature = TEMPERATURE, integrator = INTEGRATOR,
    lj_options=LJ_OPTIONS,es_options=ES_OPTIONS, minimizer = MINIM,
    minimizer_steps = MINIM_STEPS, minimizer_step_size = MINIM_STEP_SIZE)



conf1a = deepcopy(universe.configuration)
conf1b = deepcopy(universe.configuration)

traj1 = tr.Trajectory(conf1a, conf1b)

conf2a = tr.TemporalConfiguration(5.5, H1, H2, O)
conf2b = tr.TemporalConfiguration(6.5, H1, H2, O)
conf2c = tr.TemporalConfiguration(7.5, H1, H2, O)
conf2d = tr.TemporalConfiguration(8.5, H1, H2, O)

traj2 = tr.Trajectory(conf2a, conf2b, conf2c, conf2d)

t_confs = []
t_max = 2.5
for i in np.arange(0.,t_max,0.5):
    t_confs.append(tr.TemporalConfiguration(i,*universe.configuration.atom_list))

traj3 = tr.Trajectory(*t_confs)

histo1 = tr.Histogram(traj3, r = [0., 20., 0.5])
histo2 = tr.Histogram(traj3, r = [0., 20., 0.5], time = [0., t_max, 1.])

# lamp = LAMPSQw.LAMPSQw()
# file = open("/Users/thomasfarmer/Documents/QENS/Model_System_Data/Water/Bertil_Halle_data/in5_data/test2_dat_LAMP")
# lamp.parse_indep_var(file)

from tests.test_data import data
SQwfile = of.ObservableFactory.create_observable('SQw')
SQwfile.read_from_file(reader='LAMPSQw', file_name=data.READER_DATA['LAMPSQw'])

SQw = eof.ObservableFactory.create_observable('SQw')
n_Q = 10
Q_values = [2 * np.pi * i / UNIVERSE_DIMS[0] for i in range(1,n_Q)]

SQw.calculate_from_MD(traj3, Q_values = Q_values, isotropic = True, cell = np.array(UNIVERSE_DIMS))
FQt = []
for Sw in SQwfile.data[2]:
    FQt.append(np.fft.fft(Sw))


plot.plot3d_surface([0,0,np.absolute(SQw.FQt)])

w = SQwfile.data[1]
Q = SQwfile.data[0]
t = np.fft.fftfreq(w.size, w[1] - w[2])



# def timeMDMC():
#     NVESim.run(n_steps=N_STEPS)
# print timeit(timeMDMC,number=1)
#
# from MMTK.Dynamics import VelocityVerletIntegrator, VelocityScaler, \
#                           TranslationRemover
# from MMTK.Minimization import SteepestDescentMinimizer
# minimizer = SteepestDescentMinimizer(NVESim.engine.universe, step_size = 0.05*Units.Ang)
# integrator = VelocityVerletIntegrator(NVESim.engine.universe,
#                                       actions=[VelocityScaler(300., 10.),
#                                                TranslationRemover()])
#
# def timeMMTK():
#     minimizer(steps=MINIM_STEPS)
#     integrator(steps=N_STEPS)
#
# print timeit(timeMMTK,number=1)

# uni = mmtkf.MMTKCubicUniverse(universe,lj_options=1.2,es_options={'method':'ewald'})
#
# H1_mmtk = mmtkf.MMTKAtom(H1,'H')
# water_mmtk = mmtkf.MMTKMolecule(water_molecule)
#
# uni.addObject(H1_mmtk)
# uni.addObject(water_mmtk)

# universe.add_structural_unit(water_molecule)
# universe.add_force_field("SPCE")
# for interaction in universe.interaction_set():
#     print(interaction)
#     print(interaction.function)
#     print(interaction.function.params)
#     print()
# def time_fill():
#     universe.fill(water_molecule,force_field="SPCE",num_density=WATER_NUM_DENSITY)
#
# print(timeit(time_fill,number=1))
# print(len(universe.configuration))



# Performance testing for different pairwise distance algorithms - can't use fun3 as np.array is mutable

# from itertools import chain
# from copy import deepcopy
# def fun1():
#     exclude = []
#     gens = []
#     for p_o in pos:
#         exclude.append(p_o)
#         for i in pos:
#             if i not in exclude:
#                 yield (i*p_o )
#
# def fun2():
#     for i in range(len(pos)):
#         for j in range(i+1,len(pos)):
#             yield (pos[j]*pos[i])
#
# def fun3():
#     pos = set(pos)
#     s_exclude = set()
#     for p_o in pos:
#         s_exclude.update([p_o])
#         for i in pos - s_exclude:
#             yield (i*p_o)
#
import numpy as np
pos = np.random.random_sample(1000)
#
# def fun1wrap():
#     return list(fun1())
#
# def fun2wrap():
#     return list(fun2())
#
# def fun3wrap():
#     return list(fun3())
#
from timeit import timeit
# timeit(fun1wrap, number=1)
# # 5.463881969451904
# timeit(fun2wrap, number=1)
# # 0.29927802085876465
# timeit(fun3wrap, number=1)
# # 0.22504305839538574

def fun4(pos):
    for i in range(len(pos)):
        for j in range(i+1,len(pos)):
            yield (pos[j]*pos[i])

def nphistwrap():
        np.histogram(list(fun4(pos)),10)

timeit(nphistwrap,number=1)

bins = np.linspace(0,1,11)

# def fun5():
#
#
# from math import floor
# binwidth = 20
# counts = dict()
# filename = "mydata.csv"
# for val in next_value_from_file(filename):
#    binname = int(floor(val/binwidth)*binwidth)
#    if binname not in counts:
#       counts[binname] = 0
#    counts[binname] += 1
# print counts


from MMTK.Trajectory import Trajectory
import MDMC.trajectory_analysis.observables.obs_factory as of
import MDMC.MD.engine_facades.mmtk as mmtk
import numpy as np

trajectory2 = Trajectory(None, "/Users/thomasfarmer/Library/virtualenv/virtualenvMMTK/bin/MMTK-2.7.10/Simulations/Water/spce/for_MDMC/water2048_50000steps_spce.nc")
MDMC_traj2 = mmtk.convert_trajectory(trajectory2, slice=slice(50, 5010, 10))
SQw2 = of.ObservableFactory.create_observable('SQw')
n_Q = 10
Q_values2 = [2 * np.pi * i / trajectory2.universe.cellParameters()[0] for i in range(1,n_Q+1)]
SQw2.calculate_from_MD(MDMC_traj2, Q_values = Q_values2, cell = trajectory2.universe.cellParameters())
#
#
# surf = go.Surface(x = SQwfile.data[1], y = SQwfile.data[0], z = np.absolute(SQwfile[2]), colorscale = [[0, 'rgb(0,0,255)', [0.1 'rgb(128,0,128)'],[1, 'rgb(255,0,0)']])
# x = SQwfile.data[0]
# y = SQwfile.data[1]
# xGrid, yGrid = np.meshgrid(y, x)
# line_marker = dict(color='rgb(0, 0, 0)', width=2)
# for i, j, k in zip(xGrid, yGrid, z):
#     lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
#
# layout = go.Layout(
#     scene=dict(
#     xaxis=dict(
#     gridcolor='rgb(255, 255, 255)',
#     zerolinecolor='rgb(255, 255, 255)', range = [-0.3,0.3], title = 'E /meV'),
#     yaxis=dict(
#     gridcolor='rgb(255, 255, 255)',
#     zerolinecolor='rgb(255, 255, 255)', range = [0,2.25], title = 'Q /AA<sup>-1</sup>'),
#     zaxis=dict(
#     gridcolor='rgb(255, 255, 255)',
#     zerolinecolor='rgb(255, 255, 255)', title = 'S(Qw) /arb')), showlegend = False, height = 1000, width = 1000)
# fig=go.Figure(data=[surf]+lines ,layout = layout)
# py.offline.plot(fig)

# To update the trajectory file:
import cPickle as pickle
import zlib
from MMTK.Trajectory import Trajectory
import MDMC.MD.engine_facades.mmtk as mmtk
from tests.test_data import data
MMTK_trajectory = Trajectory(None, "/Users/thomasfarmer/Library/virtualenv/virtualenvMMTK/bin/MMTK-2.7.10/Simulations/Water/spce/for_MDMC/water2048_50000steps_spce.nc")
trajectory = mmtk.convert_trajectory(MMTK_trajectory, slice=slice(50, 5010, 100))
pickled_trajectory = pickle.dumps(trajectory)
compressed_trajectory = zlib.compress(pickled_trajectory)
file = open(data.OBJECT_DATA['trajectory'], 'w')
file.write(compressed_trajectory)
file.close()

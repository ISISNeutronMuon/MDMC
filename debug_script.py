"""Script for running debug"""

import MDMC.src.MD.interaction_functions as ifu
import MDMC.src.MD.structural_units as su
import MDMC.src.MD.simulation as sim
import MDMC.src.MD.force_fields as ff
import MDMC.src.MD.engine_facades.mmtk as mmtkf
import MDMC.src.trajectory_analysis.trajectory as tr
from MDMC.src.readers.LAMPSQw import LAMPSQw
import MDMC.src.trajectory_analysis.observables.obs_factory as eof
from MDMC.tests.test_data import data
from MDMC.src.utilities import plot

from timeit import timeit
from copy import deepcopy
import numpy as np


from MMTK import *
from MMTK.Minimization import SteepestDescentMinimizer


from MMTK.Trajectory import Trajectory
import MDMC.src.trajectory_analysis.observables.obs_factory as of
import MDMC.src.MD.engine_facades.mmtk as mmtk
import numpy as np

trajectory = Trajectory(None, "/Users/thomasfarmer/Library/virtualenv/virtualenvMMTK/bin/MMTK-2.7.10/Simulations/Water/spce/for_MDMC/water2048_50000steps_spce.nc")
MDMC_traj = mmtk.convert_trajectory(trajectory, slice={'start':50,'stop':5010,'step':100})
SQw = of.ObservableFactory.create_observable('SQw')
n_Q = 10
Q_values = [2 * np.pi * i / trajectory.universe.cellParameters()[0] for i in range(1,n_Q+1)]
cell = trajectory.universe.cellParameters()
SQw.calculate_from_MD(MDMC_traj, Q_values = Q_values, cell = cell, isotropic = False)


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
# universe.add_force_field(ff.SPCE)

universe.fill(water_molecule,force_field=ff.SPCE,num_density=WATER_NUM_DENSITY)

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

SQwfile = eof.ObservableFactory.create_observable('SQw')
SQwfile.read_from_file(reader='LAMPSQw', file_name=data.data['LAMPSQw'])

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
# universe.add_force_field(ff.SPCE)
# for interaction in universe.interaction_set():
#     print(interaction)
#     print(interaction.function)
#     print(interaction.function.params)
#     print()
# def time_fill():
#     universe.fill(water_molecule,force_field=ff.SPCE,num_density=WATER_NUM_DENSITY)
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
import MDMC.src.trajectory_analysis.observables.obs_factory as of
import MDMC.src.MD.engine_facades.mmtk as mmtk
import numpy as np

trajectory2 = Trajectory(None, "/Users/thomasfarmer/Library/virtualenv/virtualenvMMTK/bin/MMTK-2.7.10/Simulations/Water/spce/for_MDMC/water2048_50000steps_spce.nc")
MDMC_traj2 = mmtk.convert_trajectory(trajectory2, slice={'start':50,'stop':5010,'step':10})
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

"""Module for simple plotting, largely for testing purposes

AUTHOR :    Thomas Farmer        START DATE :    2018-6-11 14:23:28"""

import plotly as py
import plotly.graph_objs as go
import numpy as np

# TODO: Add in axes labels etc
def plot_observable(observable):

    pass


def plot2d(data):

    trace = [go.Scatter(x = data[0], y = data[1])]

    py.offline.plot(trace)


# TODO: Add in cut along other axis
def plot2d_cuts(data, n_cuts):

    cut_spacing = int(data[0].size / n_cuts)
    traces = []
    for i in range(n_cuts):
        traces.append(go.Scatter(x = data[1], y = data[2][i * cut_spacing],
            name = str(data[0][i * cut_spacing])))

    py.offline.plot(traces)


def plot3d_surface(data, log_z = False):

    x = data[0]

    y = data[1]

    if log_z:
        surf = [go.Surface(x = x, y = y, z=np.log(data[2]))]
    else:
        surf = [go.Surface(x = x, y = y, z=data[2])]

    layout = go.Layout(autosize = False, width = 1000, height = 1000)

    fig = go.Figure(data = surf, layout = layout)

    py.offline.plot(fig)


def plot_configuration(config):

    x = [position[0] for position in config]
    y = [position[1] for position in config]
    z = [position[2] for position in config]

    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
            ),
        opacity=0.8
        )
    )

    data = [trace]

    fig = go.Figure(data = data)

    py.offline.plot(fig)


# TODO: get mesh working
def plot3d_mesh(data, log_z = False):

    x = data[0]
    y = data[1]

    if log_z:
        z = [go.Surface(z=np.log(data[2]))]
    else:
        z = [go.Surface(z=data[2])]

    trace = go.Data([go.Mesh3d(x = x, y = y, z = z,
        colorscale = [['0', 'rgb(255, 0, 0)'],
        ['0', 'rgb(255, 0, 0)'],
        ['0', 'rgb(255, 0, 0)']])])

    fig = go.Figure(data = trace)

    py.offline.plot(fig)


# TODO: add ribbon plot
def plot3d_ribbon(data, log_z = False):

    raise NotImplementedError

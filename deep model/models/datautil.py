
import torch

import os
import csv
import math
import torch
import random
import zipfile
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import plotly.graph_objects as go

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.nn import TripletMarginLoss

def load_obj_from_stream(stream):
    verts = []
    faces = []

    for line in stream:
        if line == "":
            continue

        elif line.lstrip().startswith("v "):
            vertices = line.replace("\n", "").split(" ")[1:]
            verts.append(list(map(float, vertices)))

        elif line.lstrip().startswith("f "):
            t_index_list = []
            for t in line.replace("\n", "").split(" ")[2:]:
                t_index = t.split("/")[0]
                t_index_list.append(int(t_index) - 1)
            faces.append(t_index_list)

    verts = torch.tensor(verts, dtype=torch.float32, device= "cpu")
    faces = torch.tensor(faces, dtype=torch.int64, device= "cpu")
    return verts, faces

def load_obj(f):
    # from list
    if isinstance(f, list):
        return load_obj_from_stream(f)
    
    # from file
    print(f)
    with open(f, encoding='utf8') as file:
        return load_obj_from_stream(file)

def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    
    fig = go.Figure(data=data,
        layout=go.Layout(
            updatemenus=[dict(type='buttons',
                showactive=False,
                y=1,
                x=0.8,
                xanchor='left',
                yanchor='bottom',
                pad=dict(t=45, r=10),
                buttons=[dict(label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=50, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode='immediate'
                        )]
                    )
                ])]
        ),
        frames=frames
    )

    return fig

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]

    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                        line=dict(width=2,
                        color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
    
    fig.show()
    
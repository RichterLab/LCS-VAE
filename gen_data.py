#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:52:27 2020

@author: theomacmillan
"""

#%%

import numpy as np
from utils import trajectory_jet

#%% generate trajectories the vector way, jet

N_TRAIN = 300
N_SAMPLE = 50
tmax = 0.5E6
t0 = 0

yvec = np.linspace(-3E6, 3E6, N_TRAIN)
xvec = np.linspace(0, 20E6, N_TRAIN)

x0, y0 = np.meshgrid(xvec, yvec)

x0 = x0.flatten()
y0 = y0.flatten()

ts, xs, ys = trajectory_jet(x0, y0, t0, tmax, N_SAMPLE)
    
train_inputs = np.concatenate((np.transpose(xs), np.transpose(ys)))

np.save("training_data/jet_inputs.npy", train_inputs)
np.save("training_data/jet_outputs.npy", train_inputs)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:21:31 2020

@author: theomacmillan
"""

#%%

import torch
import numpy as np
from models import AE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import trajectory_jet
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

#%%

latent = 12
net = VAE(100, 100, latent, 50)
net.load_state_dict(torch.load("trained_models/VAE_turb_L12_BE2.dat"))
scaling = 20E6

#%% look at individual trajectories

N_SAMPLE = 50

t0 = 0
tmax = 0.5E6
x0 = [0]
y0 = [0]

ts, xs, ys = trajectory_turb(x0, y0, t0, tmax, N_SAMPLE)
xs = np.squeeze(xs)
ys = np.squeeze(ys)

fig, ax = plt.subplots()
ax.plot((xs)/scaling, (ys)/scaling, label='actual')
x_preds = [xs/scaling]
y_preds = [ys/scaling]


#%%

for i in range(50):
    ts, xs, ys = trajectory_turb(x0, y0, t0, tmax, N_SAMPLE)
    
    xs = np.squeeze(xs)
    ys = np.squeeze(ys)
    
    xy = np.empty(N_SAMPLE*2,)
    xy[0:N_SAMPLE]=xs
    xy[N_SAMPLE:]=ys
    
    xy = np.reshape(xy, (1, 100))/scaling
    
    x_in = torch.Tensor(xy)
    pred = net.forward(x_in).data.numpy()
    
    pred = np.squeeze(pred)
    
    x_pred = pred[0:N_SAMPLE]
    y_pred = pred[N_SAMPLE:]

    x_preds.append(x_pred)
    y_preds.append(y_pred)

    
    plt.plot(x_pred, y_pred, alpha=0.2, label='reconstructed')
plt.plot((xs)/scaling, (ys)/scaling)

#%% examine latent layers

size = 500

xrange = np.linspace(0, 20E6, size)
yrange = np.linspace(-3E6, 3E6, size)

t0 = 0
tmax = 0.5E6

x0, y0 = np.meshgrid(xrange, yrange)

x0 = x0.flatten()
y0 = y0.flatten()
ts, xs, ys = trajectory_jet(x0, y0, t0, tmax, N_SAMPLE)

xs2 = np.subtract(xs,np.expand_dims(x0,-1))
ys2 = np.subtract(ys,np.expand_dims(y0,-1))

test = np.transpose(np.concatenate((np.transpose(xs), np.transpose(ys))))/scaling

x_in = torch.Tensor(test)
results = net.forward(x_in)
latent_layer = net.mu.detach().numpy()
latent_layer2 = np.exp(net.log_sigma.detach().numpy())

#%%

n = 1 #latent node to look at

Z = np.reshape(latent_layer[:,n], (size,size))

plt.subplot(1, 1, 1)
plt.pcolor(Z)
plt.colorbar()
print(np.max(Z))

np.savetxt("activation_BE1_turb", latent_layer)










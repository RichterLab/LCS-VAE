#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:04:57 2020

@author: theomacmillan
"""
#%%

import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models import VAE
from utils import target_loss, kl_loss

#%%

latent = 12
net = VAE(100, 100, latent, 50)

inputs = np.transpose(np.load("training_data/jet_inputs.npy"))/20E6
outputs = np.transpose(np.load("training_data/jet_inputs.npy"))/20E6

#%%

beta = 0.01

inputs = torch.Tensor(inputs)
outputs = torch.Tensor(outputs)
traindata = TensorDataset(inputs, outputs)
dataloader = DataLoader(traindata, batch_size=1000, shuffle=True, num_workers=0)

SAVE_PATH = "trained_models/VAE_jet_L12_BE2.dat"

N_EPOCHS = 10000
optimizer = optim.Adam(net.parameters())
rms_loss = []
kldiv_loss = []

for epoch in range(N_EPOCHS):
    epoch_rms_loss = []
    epoch_kldiv_loss = []
    for  minibatch in dataloader:
        inputs, outputs = minibatch
        optimizer.zero_grad()
        pred = net.forward(inputs)
        kl = beta*kl_loss(net.mu, net.log_sigma)
        rms = target_loss(pred, outputs)
        loss = rms+kl
        loss.backward()
        optimizer.step()

        epoch_rms_loss.append(np.mean(rms.data.detach().numpy()))
        epoch_kldiv_loss.append(np.mean(kl.data.detach().numpy()))

    kldiv_loss.append(np.mean(epoch_kldiv_loss))
    rms_loss.append(np.mean(epoch_rms_loss))
    print("Epoch %d -- rms error %f -- kl loss %f" % 
          (epoch+1, rms_loss[-1], kldiv_loss[-1]))

torch.save(net.state_dict(), SAVE_PATH)
print("Model saved to %s" % SAVE_PATH)

np.save("kldiv_b"+str(beta), kldiv_loss)
np.save("rms_b"+str(beta), rms_loss)
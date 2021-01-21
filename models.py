#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:42:17 2020

@author: theomacmillan
"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, layer_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.enc1 = nn.Linear(input_dim, layer_dim)
        self.enc2 = nn.Linear(layer_dim, layer_dim)
        
        self.latent = nn.Linear(layer_dim, latent_dim*2)
        
        self.dec1 = nn.Linear(latent_dim, layer_dim)
        self.dec2 = nn.Linear(layer_dim, layer_dim)

        self.out = nn.Linear(layer_dim, output_dim)

    def encoder(self, x):
        z = F.relu(self.enc1(x))
        z = F.relu(self.enc2(z))
        z = F.relu(self.latent(z))

        self.mu = z[:,0:self.latent_dim]
        self.log_sigma = z[:,self.latent_dim:]
        eps = torch.randn(x.size(0), self.latent_dim)

        return self.mu+torch.exp(self.log_sigma)*eps
    
    def decoder(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))

        return self.out(x)

    def forward(self, x):
        self.latent_r = self.encoder(x)
        return self.decoder(self.latent_r)
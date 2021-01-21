#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:51:29 2020

@author: theomacmillan
"""
#%%
import torch
import numpy as np
from scipy.integrate import solve_ivp

def target_loss(pred,answer):
	return torch.mean(torch.sum((pred - answer)**2,dim=-1))

def kl_loss(mu, log_sigma):
    return torch.mean(torch.sum(-0.5*(1+log_sigma-mu.pow(2)-torch.exp(log_sigma)),dim=-1))
    

def phi(x, y, t):
    U = 62.66
    L = 1767E3
    r0 = 6371E3
    c = [0.1446*U,0.2051*U,0.4615*U]
    sigma = np.subtract(c,c[2])
    eta = [0.0075,0.15,0.3];
    s = 0
    
    for i in range(3):
        n = i+1
        kn = 2*n/r0
        s = s+eta[i]*np.cos(kn*(x-sigma[i]*t))
    
    phi0 = c[2]*y-U*L*np.tanh(y/L)
    phi1 = U*L*(1/(np.cosh(y/L)))**2*s
    phi_f = phi0 + phi1
    return phi_f

def velocity_jet(t, inp):
    x = inp[0:int(inp.shape[0]/2)]
    y = inp[int(inp.shape[0]/2):] 
    t = t*np.ones(x.shape)
    dx = 1
    dy = 1
    U = -(phi(x, y+dy, t)-phi(x, y, t))/dy
    V = (phi(x+dx, y, t)-phi(x, y, t))/dx
    vel = np.append(U,V)
    return vel

def trajectory_jet(x0, y0, t0, tmax, N_SAMPLE):
    n = len(x0)
    inp = np.append(x0, y0)
    sol = solve_ivp(velocity_jet, [t0,t0+tmax], inp, 
                    t_eval=np.linspace(t0, t0+tmax, N_SAMPLE))
    
    xs = sol.y[0:n,:]
    ys = sol.y[n:,:]
    
    return sol.t, xs, ys
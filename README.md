# LCS-VAE

This code accompanies the paper "The most robust representations of flow trajectories are Lagrangian coherent structures" by Theodore MacMillan and David H. Richter.

To run:

1. Generate trajectory data from the Bickley Jet using gen_data.py
2. Train the model on the generated data using training.py
3. Analyze data (view reconstructed trajectories and latent node activations) using analysis2.py

We credit the structure of our developed code to the repository found at https://github.com/fd17/SciNet_PyTorch which implements the variational autoencoder structure found in "Discovering physical concepts with neural networks" (Iten et. al 2020)

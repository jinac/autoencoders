"""
Vanilla Variational Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np



SCALE_CONV_FMS = [3, 64, 128]
SCALE_KERNEL_SIZE = [3, 3, 3]

def loss_fn(x, recon_x, mu, logvar, beta=1.0):
	reconst_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return reconst_loss + (beta * kl_loss)


class VAEEncoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim):
		super(Encoder, self).__init__()

		self.conv_layers = nn.ModuleList(
			nn.Conv2d(3, 32, kernel_size=4, stride=2),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
			)

		self.mu_layer = nn.Linear(hidden_dim, latent_dim)
		self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		# Convnet.
		print(x.shape)
		for layer in self.conv_layers:
			x = F.relu(layer(x))
			# print(x.shape)

		# Flatten.
		x = x.view(x.size()[0], -1)
		print(x.shape)

		# Fully connected.
		mu = self.mu_layer(x)
		logvar = self.logvar_layer(x)

		return mu, logvar


class VAEDecoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim):
		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.fc = nn.Linear(latent_dim, hidden_dim)

		self.conv_layers = nn.ModuleList(
			nn.ConvTransposed2d(hidden_dim, 128, kernel_size=5, stride=2),
			nn.ConvTranspose2d(128, 64, kenrnel_size=5, stride=2),
			nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
			nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
			)

	def forward(self, z):
		z = x.view(input.size(0), self.hidden_dim, 1, 1)

		for layer in self.conv_layers[:-1]:
			z = F.relu(layer(x))

		z = F.sigmoid(layer(x))

		return z


class VAE(nn.Module):
	def __init__(self, latent_dim, hidden_dim):
		self.encoder = VAEEncoder(latent_dim, hidden_dim)
		self.decoder = VAEDecoder(latent_dim, hidden_dim)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + (eps * std)

	def encode(self, x):
		mu, logvar = self.encoder.forward(x)
		z = self.reparameterize(mu, logvar)
		return z

	def decode(self, z):
		return self.decoder.forward(z)

	def forward(self, x):
		mu, logvar = self.encoder.forward(x)
		z = self.reparameterize(mu, logvar)
		return self.decoder.forward(z), mu, logvar

"""
Adversarial Autoencoder

Using ideas from https://arxiv.org/pdf/1511.05644.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def loss_fn(x, x_reconst, mu, logvar, beta=1.0):
	reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
	kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return reconst_loss + (beta * kl_loss)


class Encoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim):
		super(Encoder, self).__init__()

		self.conv_layers = nn.ModuleList([
			nn.Conv2d(3, 32, kernel_size=4, stride=2),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
		])

		self.fc_layer = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		# Convnet.
		for layer in self.conv_layers:
			x = F.relu(layer(x))

		# Flatten.
		x = x.view(x.size()[0], -1)

		# Fully connected.
		z = self.fc_layer(x)

		return z


class Decoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim):
		super(Decoder, self).__init__()

		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.fc_layer = nn.Linear(latent_dim, hidden_dim)

		self.conv_layers = nn.ModuleList([
			nn.ConvTranspose2d(hidden_dim, 128, kernel_size=5, stride=2),
			nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
			nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
			nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
		])

	def forward(self, z):
		# FC from latent to hidden dim.
		z = F.relu(self.fc_layer(z))

		# Unflatten.
		z = z.view(z.size(0), self.hidden_dim, 1, 1)

		# Convnet.
		for layer in self.conv_layers[:-1]:
			z = F.relu(layer(z))
		z = F.sigmoid(self.conv_layers[-1](z))

		return z


class Critic(nn.Module):
	def __init__(self, latent_dim):
		super(Critic, self).__init__()

		self.latent_dim = latent_dim
		self.fc_layers = nn.ModuleList([
			nn.Linear(self.latent_dim, 64),
			nn.Linear(64, 32),
			nn.Linear(32, 32),
		])

	def forward(self, z):
		for layer in self.fc_layers[:-1]:
			z = F.leaky_relu(layer(z), 0.2)
		z = F.sigmoid(self.fc_layers[-1](z))

		return z


class AAE(nn.Module):
	def __init__(self, latent_dim, hidden_dim):
		super(VAE, self).__init__()

		self.encoder = Encoder(latent_dim, hidden_dim)
		self.decoder = Decoder(latent_dim, hidden_dim)
		self.critic = Critic(latent_dim)

	def encode(self, x):
		mu, logvar = self.encoder.forward(x)
		z = self.reparameterize(mu, logvar)
		return z

	def decode(self, z):
		return self.decoder.forward(z)

	def forward(self, x):
		z = self.encoder.forward(x)
		return self.decoder.forward(z)

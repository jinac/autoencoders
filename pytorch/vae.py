"""
Vanilla Variational Autoencoder
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

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Convnet.
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        # Flatten.
        x = x.view(x.size()[0], -1)

        # Fully connected.
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        return mu, logvar


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


class VAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)

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

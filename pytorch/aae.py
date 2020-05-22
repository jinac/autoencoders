"""
Adversarial Autoencoder

Using ideas from https://arxiv.org/pdf/1511.05644.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def loss_fn(x, x_reconst, critic_out, ones, beta=1.0):
    reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    print(critic_out.shape, ones.shape)
    critic_loss = F.binary_cross_entropy(critic_out, ones, reduction='sum')
    return reconst_loss + critic_loss


class Encoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Convnet.
        x = self.conv_net(x)


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

        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # FC from latent to hidden dim.
        z = F.relu(self.fc_layer(z))

        # Unflatten.
        z = z.view(z.size(0), self.hidden_dim, 1, 1)

        # Convnet.
        z = self.deconv_net(z)

        return z


class Critic(nn.Module):
    def __init__(self, latent_dim):
        super(Critic, self).__init__()

        self.latent_dim = latent_dim
        self.critic = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.critic(z)


class AAE(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(AAE, self).__init__()

        self.encoder = Encoder(latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim)
        self.critic = Critic(latent_dim)

    def gan_fake_forward(self, z):
        return self.critic.forward(z)

    def forward(self, x):
        z = self.encoder.forward(x)
        return self.decoder.forward(z), self.critic.forward(z)

"""
WGAN

See:
https://github.com/tolstikhin/wae/blob/master/wae.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def loss_fn(critic_out, labels):
    critic_loss = F.binary_cross_entropy(critic_out, labels, reduction='sum')
    return critic_loss


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.fc_layer = nn.Linear(latent_dim, hidden_dim)
        self.fc_bn = nn.BatchNorm1d(hidden_dim)

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
        z = F.relu(self.fc_bn(self.fc_layer(z)))

        # Unflatten.
        z = z.view(z.size(0), self.hidden_dim, 1, 1)

        # Convnet.
        z = self.deconv_net(z)

        return z


class Critic(nn.Module):
    def __init__(self, latent_dim):
        super(Critic, self).__init__()

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

        self.lin_layers = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        # Convnet.
        x = self.conv_net(z)

        # Flatten.
        x = x.view(x.size()[0], -1)

        # Linear layers for discrimation.
        x = self.lin_layers(x)

        return x


class WGAN(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(WGAN, self).__init__()

        self.generator = Generator(latent_dim, hidden_dim)
        self.critic = Critic(latent_dim)

    def forward(self, z):
        x = self.generator.forward(z)
        return self.critic.forward(x)

"""
VQ-VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(x, x_reconst, embed_loss):
	reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
	return reconst_loss + embed_loss


# Copied from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class Encoder(nn.Module):
	def __init__(self, latent_dim, hidden_dim):
		super(Encoder, self).__init__()

		self.conv_layers = nn.ModuleList([
			nn.Conv2d(3, 32, kernel_size=4, stride=2),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
		])

		self.lin_layer = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		# Convnet.
		# print(x.shape)
		for layer in self.conv_layers:
			x = F.relu(layer(x))
			# print(x.shape)

		# Flatten.
		x = x.view(x.size()[0], -1)
		# print(x.shape)

		# Fully connected.
		return self.lin_layer(x)


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


class VQVAE(nn.Module):
	def __init__(self, latent_dim, hidden_dim, codebook_size, beta=0.25):
		super(VQVAE, self).__init__()

		self.encoder = Encoder(latent_dim, hidden_dim)
		self.decoder = Decoder(latent_dim, hidden_dim)
		self.quantizer = VectorQuantizer(codebook_size, latent_dim, beta)
		
	def encode(self, x):
		mu, logvar = self.encoder.forward(x)
		z = self.reparameterize(mu, logvar)
		return z

	def decode(self, z):
		return self.decoder.forward(z)

	def forward(self, x):
		z = self.encoder.forward(x)
		embed_loss, z_q, perplexity, _, _ = self.quantizer.forward(z)
		return self.decoder.forward(z), embed_loss, perplexity

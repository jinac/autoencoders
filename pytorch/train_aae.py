"""
Training script for aae.
"""
import os

import numpy as np
import torch.optim as optim
import torch

import data_util
import aae


def main():
	# Set variables.
	img_dim = [64, 64]
	latent_dim = 32
	hidden_dim = 1024
	num_epochs = 5000
	batch_size = 64
	shuffle = True
	num_loader_workers = 2
	beta = 1.0
	std_dev = 1.
	mu = 0.
	cuda = True
	save_dir = os.path.dirname(os.path.realpath(__file__))

	# Load Encoder, Decoder.
	aae_net = aae.AAE(latent_dim, hidden_dim)
	if cuda:
		aae_net.cuda()

	# Set loss fn.
	loss_fn = aae.loss_fn

	# Load optimizer.
	optimizer = optim.Adam(aae_net.parameters(), lr=0.0002)

	# Load Dataset.
	anime_data = data_util.AnimeFaceData(img_dim, batch_size, shuffle, num_loader_workers)

	# Epoch loop
	ones = torch.Tensor(np.ones(batch_size))
	if cuda:
		ones = ones.cuda()
	zeroes = torch.Tensor(np.zeros(batch_size))
	if cuda:
		zeroes = zeroes.cuda()

	for epoch in range(num_epochs):
		print('Epoch {} of {}'.format(epoch + 1, num_epochs))

		# Batch loop.
		for i_batch, batch_data in enumerate(anime_data.data_loader, 0):
			print('Batch {}'.format(i_batch+1))

			# Load batch.
			x, _ = batch_data
			if cuda:
				x = x.cuda()

			# Reset gradient.
			optimizer.zero_grad()

			# Run batch, calculate loss, and backprop.
			# Train autoencoder and gan on real batch.
			x_reconst, real_critic = aae_net.forward(x)
			loss = loss_fn(x, x_reconst, real_critic, ones, beta)
			loss.backward()
			optimizer.step()

			# Train gan on fake batch.
			fake_z = std_dev * np.random.randn(batch_size, latent_dim) + mu
			fake_critic = aae_net.gan_fake_forward(fake_z)
			loss = F.binary_cross_entropy(critic_out, zeros, reduction='sum')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch % 500 == 0:
			util.save_weights(vae_net, os.join(save_dir, 'aae_{}.pth'.format(epoch)))


if __name__ == '__main__':
	main()

"""
Training script for vae, beta-vae.
"""
import torch.optim as optim

import data_util
import vae


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
	cuda = True

	# Load Encoder, Decoder.
	vae_net = vae.VAE(latent_dim, hidden_dim)
	if cuda:
		vae_net.cuda()

	# Set loss fn.
	loss_fn = vae.loss_fn

	# Load optimizer.
	optimizer = optim.Adam(vae_net.parameters(), lr=0.0002)

	# Load Dataset.
	anime_data = data_util.AnimeFaceData(img_dim, batch_size, shuffle, num_loader_workers)

	# Epoch loop
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
			x_reconst, mu, logvar = vae_net.forward(x)
			loss = loss_fn(x, x_reconst, mu, logvar, beta)
			loss.backward()
			optimizer.step()


if __name__ == '__main__':
	main()

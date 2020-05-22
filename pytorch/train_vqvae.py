"""
Training script for vae, beta-vae.
"""
import torch.optim as optim

import data_util
import vqvae


def main():
	# Set variables.
	img_dim = [64, 64]
	codebook_size = 256
	latent_dim = 32
	hidden_dim = 1024
	num_epochs = 5000
	batch_size = 64
	shuffle = True
	num_loader_workers = 4
	beta = 1.0
	cuda = True

	# Load Encoder, Decoder.
	model_net = vqvae.VQVAE(latent_dim, hidden_dim, codebook_size)
	if cuda:
		model_net.cuda()

	# Set loss fn.
	loss_fn = vqvae.loss_fn

	# Load optimizer.
	optimizer = optim.Adam(model_net.parameters(), lr=0.0002)

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
			x_reconst, embed_loss, _ = model_net.forward(x)
			loss = loss_fn(x, x_reconst, embed_loss)
			loss.backward()
			optimizer.step()


if __name__ == '__main__':
	main()

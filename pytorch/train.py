"""
Training script for vae, beta-vae.
"""
import torch.optim as optim

import vae


def main():
	# Set variables.
	num_epochs = 5000
	beta = 1.0

	# Load Encoder, Decoder.
	vae_net = vae.VAE(32, 1024)

	# Set loss fn.
	loss_fn = vae.loss_fn

	# Load optimizer.
	optimizer = optim.Adam(encoder.parameters(), lr=0.0002)

	# Load Dataset.

	# Training loop
	for epoch in range(epochs):
		# Reset gradient.
		optimizer.zero_grad()

		# Load batch.
		x = None

		# Run batch, calculate loss, and backprop.
		x_reconst, mu, logvar = vae_net.forward(x)
		loss = loss_fn(x, x_reconst, mu, logvar, beta)
		loss.backward()
		optimizer.step()


if __name__ == '__main__':
	main()

"""
Training script for aae.
"""
import os

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch

import aae
import data_util
import util


def main():
    # Set variables.
    img_dim = [64, 64]
    latent_dim = 32
    hidden_dim = 1024
    num_epochs = 100
    save_freq = 25
    batch_size = 64
    shuffle = True
    num_loader_workers = 2
    std_dev = 1.
    mu = 0.
    cuda = True
    learning_rate = 0.001
    save_dir = os.path.dirname(os.path.realpath(__file__))

    # fix seed for experiment.
    util.fix_seed()

    # Load Encoder, Decoder.
    aae_net = aae.AAE(latent_dim, hidden_dim)
    if cuda:
        aae_net.cuda()

    # Set loss fn.
    loss_fn = aae.loss_fn

    # Load optimizer.
    optimizer = optim.Adam(aae_net.parameters(), lr=learning_rate)

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
            loss = loss_fn(x, x_reconst, real_critic, ones)
            loss.backward()
            optimizer.step()

            # Train gan on fake batch.
            fake_z = torch.Tensor(std_dev * np.random.randn(batch_size, latent_dim) + mu)
            if cuda:
                fake_z = fake_z.cuda()
            fake_critic = aae_net.gan_fake_forward(fake_z)
            loss = F.binary_cross_entropy(fake_critic, zeroes, reduction='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % save_freq == 0:
            util.save_weights(vae_net, os.path.join(save_dir, 'aae_{}.pth'.format(epoch)))

        end = time.time()
        print('loss: ', loss)
        print('Took {}'.format(end - start))

if __name__ == '__main__':
    main()

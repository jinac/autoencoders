"""
Training script for vae, beta-vae.
"""
import os
import time

import torch
import torch.optim as optim

import data_util
import util
import vae


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
    beta = 1.
    cuda = True
    learning_rate = 0.01
    save_dir = os.path.dirname(os.path.realpath(__file__))

    # fix seed for experiment.
    util.fix_seed()

    # Load Encoder, Decoder.
    vae_net = vae.VAE(latent_dim, hidden_dim)
    if cuda:
        vae_net.cuda()

    # Set loss fn.
    loss_fn = vae.loss_fn

    # Load optimizer.
    optimizer = optim.Adam(vae_net.parameters(), lr=learning_rate)

    # Load Dataset.
    anime_data = data_util.AnimeFaceData(img_dim, batch_size, shuffle, num_loader_workers)

    # Epoch loop
    for epoch in range(1, num_epochs+1):
        print('Epoch {} of {}'.format(epoch, num_epochs))
        start = time.time()
        train_loss = 0

        # Batch loop.
        for i_batch, batch_data in enumerate(anime_data.data_loader, 0):
            # print('Batch {}'.format(i_batch+1))

            # Load batch.
            x, _ = batch_data
            if cuda:
                x = x.cuda()

            # Reset gradient.
            optimizer.zero_grad()

            # Run batch, calculate loss, and backprop.
            x_reconst, mu, logvar = vae_net.forward(x)
            loss = loss_fn(x, x_reconst, mu, logvar, beta)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        if epoch % save_freq == 0:
            util.save_weights(vae_net, os.path.join(save_dir, 'vae_{}.pth'.format(epoch)))

        end = time.time()
        print('loss: ', train_loss / len(anime_data.img_folder))
        print('Took {}'.format(end - start))


if __name__ == '__main__':
    main()

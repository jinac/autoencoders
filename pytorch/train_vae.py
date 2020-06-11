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
    num_epochs = 20
    save_freq = 5
    batch_size = 128
    shuffle = True
    num_loader_workers = 3
    beta = 1.
    cuda = True
    learning_rate = 0.001
    adaptive = False  # True
    save_dir = os.path.dirname(os.path.realpath(__file__))

    # fix seed for experiment.
    util.fix_seed()

    # Load Encoder, Decoder.
    vae_net = vae.VAE(latent_dim, hidden_dim)
    if cuda:
        vae_net.cuda()

    # Set loss fn.
    loss_fn = vae.loss_fn

    # Load Dataset.
    anime_data = data_util.AnimeFaceData(img_dim, batch_size, shuffle, num_loader_workers)

    # Load optimizer.
    if adaptive:
        optimizer = optim.Adam(vae_net.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(vae_net.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1,
                                                  epochs=num_epochs,
                                                  steps_per_epoch=10)

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
            if not adaptive:
                scheduler.step()

        if epoch % save_freq == 0:
            if adaptive:
                o = 'adaptive'
            else:
                o = 'cyclic'
            util.save_weights(vae_net, os.path.join(save_dir, 'vae_{}_{}.pth'.format(o, epoch)))

        end = time.time()
        print('loss: ', train_loss / len(anime_data.img_folder))
        print('Took {}'.format(end - start))

        if adaptive:
            o = 'adaptive'
        else:
            o = 'cyclic'
        util.save_weights(vae_net, os.path.join(save_dir, 'vae_{}_{}.pth'.format(o, epoch)))


if __name__ == '__main__':
    main()

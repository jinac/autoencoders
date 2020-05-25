"""
Training script for wgan.
"""
import os

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch

import data_util
import wgan2


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
    train_ratio = 5
    cuda = True
    clamp_low, clamp_high = -0.01, 0.01
    save_dir = os.path.dirname(os.path.realpath(__file__))

    # Load Encoder, Decoder.
    wgan_net = wgan2.WGAN(latent_dim, hidden_dim)
    if cuda:
        wgan_net.cuda()

    # Set loss fn.
    loss_fn = wgan2.loss_fn

    # Load optimizer.
    d_optimizer = optim.Adam(wgan_net.critic.parameters(), lr=0.0001)
    g_optimizer = optim.Adam(wgan_net.generator.parameters(), lr=0.0001)

    # Load Dataset.
    anime_data = data_util.AnimeFaceData(img_dim, batch_size, shuffle, num_loader_workers)

    # Epoch loop
    pos_labels = torch.Tensor(np.ones(batch_size))
    if cuda:
        pos_labels = pos_labels.cuda()
    neg_labels = torch.Tensor(-1 * np.ones(batch_size))
    if cuda:
        neg_labels = neg_labels.cuda()

    for epoch in range(num_epochs):
        print('Epoch {} of {}'.format(epoch + 1, num_epochs))

        # Batch loop.
        for i_batch, batch_data in enumerate(anime_data.data_loader, 0):
            print('Batch {}'.format(i_batch+1))

            # Load batch.
            x, _ = batch_data
            if cuda:
                x = x.cuda()

            for _ in range(train_ratio):
                # Train fake batch.
                z_fake = np.random.normal(0., 1., [batch_size, self.in_dim])
                if cuda:
                    z_fake = z_fake.cuda()
                out_fake = wgan_net.forward(z_fake)
                loss = loss_fn(out_fake, pos_labels)
                d_optimizer.zero_grad()
                loss.backward()
                d_optimizer.step()

                # Train real batch.
                out_real = wgan_net.critic.forward(x)
                loss = loss_fn(out_real, neg_labels)
                d_optimizer.zero_grad()
                loss.backward()
                d_optimizer.step()

            # Train generator .
            z_fake = np.random.normal(0., 1., [batch_size, self.in_dim])
            if cuda:
                z_fake = z_fake.cuda()
            out_fake = wgan_net.generator.forward(z_fake)
            loss = loss_fn(out_fake, pos_labels)
            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()

        if epoch % 500 == 0:
            util.save_weights(vae_net, os.join(save_dir, 'aae_{}.pth'.format(epoch)))


if __name__ == '__main__':
    main()

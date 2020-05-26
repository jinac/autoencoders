"""
Training script for vqvae
"""
import os

import torch.optim as optim

import data_util
import util
import vqvae


def main():
    # Set variables.
    img_dim = [64, 64]
    codebook_size = 256
    latent_dim = 32
    hidden_dim = 1024
    num_epochs = 100
    save_freq = 25
    batch_size = 64
    shuffle = True
    num_loader_workers = 4
    beta = 1.0
    cuda = True
    learning_rate = 0.001
    save_dir = os.path.dirname(os.path.realpath(__file__))

    # fix seed for experiment.
    util.fix_seed()

    # Load Encoder, Decoder.
    model_net = vqvae.VQVAE(latent_dim, hidden_dim, codebook_size)
    if cuda:
        model_net.cuda()

    # Set loss fn.
    loss_fn = vqvae.loss_fn

    # Load optimizer.
    optimizer = optim.Adam(model_net.parameters(), lr=learning_rate)

    # Load Dataset.
    anime_data = data_util.AnimeFaceData(img_dim, batch_size, shuffle, num_loader_workers)

    # Epoch loop
    for epoch in range(num_epochs):
        print('Epoch {} of {}'.format(epoch, num_epochs))
        start = time.time()

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

        if epoch % save_freq == 0:
            util.save_weights(vae_net, os.path.join(save_dir, 'vqvae_{}.pth'.format(epoch)))

        end = time.time()
        print('loss: ', train_loss / len(anime_data.img_folder))
        print('Took {}'.format(end - start))

if __name__ == '__main__':
    main()

"""
Implementation of GAN
"""
from tensorflow.keras import backend
from tensorflow.keras.layers import (Input, Activation,
                                     Conv2DTranspose,
                                     Dense, Dropout,
                                     Flatten, LeakyReLU,
                                     Reshape)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Progbar

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

from blocks import (bn_dense,
                    bn_conv_layer,
                    bn_deconv_layer)

class GAN(object):
    def __init__(self, input_dim, img_dim,
                 g_hidden=16,
                 d_hidden=16,
                 gen_opt=Adam,
                 gen_learning_rate=0.0001,
                 critic_opt=Adam,
                 critic_learning_rate=0.0001):
        # Define IO dimensions for models.
        self.in_dim = input_dim
        self.img_dim = img_dim

        # Hidden nodes.
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden

        # Optimizer settings.
        self.gen_lr = gen_learning_rate
        self.gen_opt = gen_opt
        self.critic_lr = critic_learning_rate
        self.critic_opt = critic_opt

        # Component networks.
        self.generator = self._construct_generator()
        self.critic = self._construct_critic()
        self.gan = self._construct_gan()

    def _construct_generator(self):
        z = Input(shape=(self.in_dim,))
        # z0 = bn_dense(z, self.g_hidden * 64 * 4 * 4, 'selu')
        z0 = Reshape((1, 1, self.in_dim))(z)
        z1 = bn_deconv_layer(
            z0, 16 * self.g_hidden, 4, 4, activation='selu', batchnorm=False)
        # z2 = bn_deconv_layer(
        #     z1, 32 * self.g_hidden, 4, 2, activation='selu', batchnorm=False)
        # z3 = bn_deconv_layer(
        #     z1, 16 * self.g_hidden, 4, 2, activation='selu', batchnorm=False)
        z4 = bn_deconv_layer(
            z1, 8 * self.g_hidden, 4, 2, activation='selu', batchnorm=False)
        z5 = bn_deconv_layer(
            z4, 4 * self.g_hidden, 4, 2, activation='selu', batchnorm=False)
        z6 = bn_deconv_layer(
            z5, 2 * self.g_hidden, 4, 2, activation='selu', batchnorm=False)
        gen_img = bn_deconv_layer(
            z6, self.img_dim[-1], 4, 2, activation='sigmoid', batchnorm=False)

        generator = Model(z, gen_img)
        # generator.compile(optimizer=self.critic_opt(lr=self.critic_lr),
        #                   loss='binary_crossentropy')
        return generator

    def _construct_critic(self):
        img = Input(shape=self.img_dim)
        d1 = bn_conv_layer(
            img, self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        d2 = bn_conv_layer(
            d1, self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        d3 = bn_conv_layer(
            d2, 2 * self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        d4 = bn_conv_layer(
            d3, 4 * self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        d5 = bn_conv_layer(
            d4, 8 * self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        # d6 = bn_conv_layer(
        #     d5, 16 * self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        # d7 = bn_conv_layer(
        #     d6, 32 * self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        # d8 = bn_conv_layer(
        #     d7, 64 * self.d_hidden, 4, 2, activation='selu', batchnorm=False)
        d_flat = Flatten()(d5)
        d_dense = bn_dense(d_flat, 1024)
        disc_out = bn_dense(d_dense, 1, activation='sigmoid')

        critic = Model(img, disc_out)
        critic.compile(optimizer=self.critic_opt(lr=self.critic_lr),
                       loss='binary_crossentropy')
        return critic

    def _construct_gan(self):
        self.critic.trainable = False
        gan = Model(self.generator.inputs[0],
                    self.critic(self.generator.output))
        gan.compile(optimizer=self.gen_opt(lr=self.gen_lr),
                    loss='binary_crossentropy')
        return gan

    def _prep_fake(self, batch_size):
        z_fake = np.random.normal(0., 1., [batch_size, self.in_dim])
        x_fake = self.generator.predict_on_batch(z_fake)
        return z_fake, x_fake

    def train_on_batch(self, x_train):
        batch_size = x_train.shape[0]
        half_batch = batch_size / 2
        idx = np.random.randint(0, batch_size, half_batch)

        # 1. Train Critic (discriminator).
        # Train on real.
        critic_loss = self.critic.train_on_batch(x_train[idx], np.ones(half_batch))
        # Train on fakes.
        z_fake, x_fake = self._prep_fake(half_batch)
        critic_loss += self.critic.train_on_batch(x_fake, np.zeros(half_batch))
        critic_loss *= 0.5

        # 2. Train Generator.
        z_fake = np.random.normal(0., 1., [batch_size, self.in_dim])
        gen_loss = self.gan.train_on_batch(z_fake, np.ones(batch_size))

        sum_loss = critic_loss + gen_loss
        return (sum_loss, critic_loss, gen_loss)


def save_grid(fname, generator, z_sample):
    cell_size = (64, 64, 3)
    n = 15

    grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
    grid_y = norm.ppf(np.linspace(0.01, 0.99, n))
    figure = np.zeros([cell_size[0] * n,
                       cell_size[1] * n,
                       cell_size[2]])
    x_decoded = generator.predict(z_sample)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            cell = x_decoded[i * n + j].reshape(*cell_size)
            figure[i * cell_size[0]: (i + 1) * cell_size[0],
                   j * cell_size[1]: (j + 1) * cell_size[1],
                   :] = cell

    plt.figure(figsize=(10, 10))
    # plt.imshow(figure)
    plt.imsave(fname, figure)
    plt.close()


def main():
    train_dir = '/mnt/scratch/code/data/animeface_subset/'
    prefix = 'animegan_1'
    num_samples = 8200
    batch_size = 100

    # train_dir = '/mnt/scratch/code/data/anime_subset/'
    # prefix = 'anime_subset_1'
    # num_samples = 2252
    # batch_size = 4

    num_epochs = 100
    # num_epochs = 5
    img_size = (64, 64, 3)
    z_dim = 100
    cmode = 'rgb'
    num_batches = num_samples / batch_size


    img_dg = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_data = img_dg.flow_from_directory(train_dir,
                                            color_mode=cmode,
                                            target_size=img_size[:2],
                                            batch_size=batch_size,
                                            class_mode=None)

    gan = GAN(z_dim, img_size)
    print(gan.generator.summary())
    print(gan.critic.summary())
    print(gan.gan.summary())

    z_sample = np.random.normal(size=[15**2, z_dim]).astype('float32')

    for epoch in xrange(num_epochs):
        print('Epoch {} of {}'.format(epoch + 1, num_epochs))
        progress_bar = Progbar(target=num_batches)

        epoch_critic_loss = []
        epoch_gen_loss = []
        epoch_sum_loss = []

        for batch_idx in xrange(num_batches):
            progress_bar.update(batch_idx)
            batch_xs = train_data.next()

            sum_loss, critic_loss, gen_loss = gan.train_on_batch(batch_xs)
            
            epoch_sum_loss.append(sum_loss)
            epoch_critic_loss.append(critic_loss)
            epoch_gen_loss.append(gen_loss)

        print('')
        print('gan loss: ', np.mean(np.array(epoch_sum_loss), axis=0))
        print('generator loss: ', np.mean(np.array(epoch_gen_loss), axis=0))
        print('discriminator loss: ', np.mean(np.array(epoch_critic_loss), axis=0))

        # Save sample
        if (epoch % 10) == 0:
            save_grid('{}_epoch_{}.png'.format(prefix, epoch),
                      gan.generator, z_sample)

    # Save model.
    with open(prefix + '_generator.json', 'w') as json_file:
        json_file.write(gan.generator.to_json())
    gan.generator.save_weights(prefix + '_generator.h5')
    save_grid('{}_epoch_{}.png'.format(prefix, epoch),
              gan.generator,
              z_sample)


if __name__ == '__main__':
    main()
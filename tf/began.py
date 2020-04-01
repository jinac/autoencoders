"""
Implementation of BEGAN
"""

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Concatenate,
                          Dense, Dropout,
                          Flatten, Reshape,
                          Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

from blocks import (bn_dense,
                    bn_deconv_layer,
                    convnet,
                    deconvnet)


class DiscriminatorLoss(object):
    """
    Taken from:
    https://github.com/mokemokechicken/keras_BEGAN/blob/master/src/began/training.py
    """
    __name__ = 'discriminator_loss'

    def __init__(self, initial_k=0, lambda_k=0.001, gamma=0.5):
        self.lambda_k = lambda_k
        self.gamma = gamma
        self.k_var = K.variable(initial_k, dtype=K.floatx(), name="discriminator_k")
        self.m_global_var = K.variable(0, dtype=K.floatx(), name="m_global")
        self.loss_real_x_var = K.variable(0, name="loss_real_x")  # for observation
        self.loss_gen_x_var = K.variable(0, name="loss_gen_x")    # for observation
        self.updates = []

    def __call__(self, y_true, y_pred):  # y_true, y_pred shape: (BS, row, col, ch * 2)
        data_true, generator_true = y_true[:, :, :, 0:3], y_true[:, :, :, 3:6]
        data_pred, generator_pred = y_pred[:, :, :, 0:3], y_pred[:, :, :, 3:6]
        loss_data = K.mean(K.abs(data_true - data_pred), axis=[1, 2, 3])
        loss_generator = K.mean(K.abs(generator_true - generator_pred), axis=[1, 2, 3])
        ret = loss_data - self.k_var * loss_generator

        # for updating values in each epoch, use `updates` mechanism
        # DiscriminatorModel collects Loss Function's updates attributes
        mean_loss_data = K.mean(loss_data)
        mean_loss_gen = K.mean(loss_generator)

        # update K
        new_k = self.k_var + self.lambda_k * (self.gamma * mean_loss_data - mean_loss_gen)
        new_k = K.clip(new_k, 0, 1)
        self.updates.append(K.update(self.k_var, new_k))

        # calculate M-Global
        m_global = mean_loss_data + K.abs(self.gamma * mean_loss_data - mean_loss_gen)
        self.updates.append(K.update(self.m_global_var, m_global))

        # let loss_real_x mean_loss_data
        self.updates.append(K.update(self.loss_real_x_var, mean_loss_data))

        # let loss_gen_x mean_loss_gen
        self.updates.append(K.update(self.loss_gen_x_var, mean_loss_gen))

        return ret

    @property
    def k(self):
        return K.get_value(self.k_var)

    @property
    def m_global(self):
        return K.get_value(self.m_global_var)

    @property
    def loss_real_x(self):
        return K.get_value(self.loss_real_x_var)

    @property
    def loss_gen_x(self):
        return K.get_value(self.loss_gen_x_var)


class BEGAN(object):
    def __init__(self, latent_dim, img_dim,
                 initial_lr=0.0001,
                 min_lr=0.00001,
                 lr_decay_rate=0.9,
                 ae_opt=Adam,
                 gen_opt=Adam,
                 enc_param=16,
                 dec_param=16,
                 initial_k=0):
        # Define IO dimensions for models.
        self.latent_dim = latent_dim
        self.img_dim = img_dim

        # Misc variables.
        self.enc_param = enc_param
        self.dec_param = dec_param
        self.ae_opt = ae_opt
        self.gen_opt = gen_opt

        # Epoch state vars.
        self.initial_k = initial_k
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.min_lr = min_lr
        self.lr_decay_step = 0
        self.m_global = np.Inf
        self.m_global_history = []

        # Component networks.
        self.encoder = self._construct_encoder()
        self.decoder = self._construct_imggen()
        self.autoencoder = self._assemble_autoencoder()
        (self.discriminator,
         self.disc_loss) = self._assemble_discriminator()

        self.generator = self._assemble_generator()

    def _construct_imggen(self):
        """
        CNN decoder/generator.

        The decoder and generator share structure but do NOT share weights.
        """
        z = Input(shape=(self.latent_dim,))
        z0 = Dense(512)(z)
        deconv_block = deconvnet(z0, self.img_dim, self.dec_param,
                                 activation='elu')
        gen_img = bn_deconv_layer(deconv_block, self.img_dim[-1], 4, 2,
                                  activation='sigmoid', batchnorm=False)

        generator = Model(z, gen_img)
        return generator

    def _construct_encoder(self):
        """
        CNN encoder.
        """
        img = Input(shape=self.img_dim)
        conv_block = convnet(img, self.enc_param, activation='elu')
        flat_1 = Flatten()(conv_block)
        fc_1 = bn_dense(flat_1, 512)
        z = Dense(self.latent_dim)(fc_1)

        encoder = Model(img, z)
        return encoder

    def _assemble_autoencoder(self):
        """
        Autoencoder.
        """
        autoencoder = Model(self.encoder.input,
                            self.decoder(self.encoder.output))
        autoencoder.compile(optimizer=self.ae_opt(lr=self.initial_lr),
                            loss='mse')
        return autoencoder

    def _assemble_discriminator(self):
        """
        Discriminator.
        """
        real_in = Input(shape=self.img_dim)
        real_out = self.autoencoder(real_in)

        fake_in = Input(shape=self.img_dim)
        fake_out = self.autoencoder(fake_in)

        out = Concatenate(axis=-1)([real_out, fake_out])

        critic = Model([real_in, fake_in], out)
        disc_loss = DiscriminatorLoss(self.initial_k)

        critic.compile(optimizer=self.ae_opt(lr=self.initial_lr),
                       loss=disc_loss)

        return critic, disc_loss

    def _assemble_generator(self):
        """
        """
        generator = self._construct_imggen()

        def generator_loss(y_true, y_pred):
            y_pred_reconst = self.autoencoder(y_pred)
            return K.mean(K.abs(y_pred - y_pred_reconst), axis=[1, 2, 3])

        generator.compile(optimizer=self.gen_opt(self.initial_lr),
                          loss=generator_loss)
        return generator


    def train_on_batch(self, x_train):
        """
        One gradient training on input batch. 
        """
        # Prep data for batch update.
        batch_size = x_train.shape[0]
        z_rand = np.random.normal(0, 1, (batch_size, self.latent_dim))
        z_img = self.generator.predict_on_batch(z_rand)
        x_disc = [x_train, z_img]

        # 1. Train Discriminator.
        disc_loss = self.discriminator.train_on_batch(x_disc,
                                                      np.concatenate(x_disc,
                                                                     axis=-1))

        # 2. Train Generator.
        z_rand = np.random.normal(0, 1, (batch_size, self.latent_dim))
        gen_loss = self.generator.train_on_batch(z_rand, np.zeros_like(z_img))

        # Update epoch state vars.
        self.m_global_history.append(self.disc_loss.m_global)

        return disc_loss + gen_loss, disc_loss, gen_loss


    def reset_epoch(self):
        epoch_m_global = np.average(self.m_global_history)
        if self.m_global <= epoch_m_global:  # decay learning rate.
            self.lr_decay_step += 1
        self.m_global = epoch_m_global

        lr = max(self.initial_lr * (self.lr_decay_rate ** self.lr_decay_step), self.min_lr)
        K.set_value(self.generator.optimizer.lr, lr)
        K.set_value(self.discriminator.optimizer.lr, lr)
        self.m_global_history = []

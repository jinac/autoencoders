"""
"""
import math

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np


def get_data_iter(data_dir, batch_size=32, target_size=(64, 64, 3)):
    if target_size[-1] == 3:
        cmode = 'rgb'
    else:
        cmode = 'grayscale'

    img_dg = ImageDataGenerator(rescale=1./255,
                                horizontal_flip=False)
    train_data = img_dg.flow_from_directory(data_dir,
                                            color_mode=cmode,
                                            target_size=target_size[:2],
                                            batch_size=batch_size,
                                            class_mode=None)

    num_batches = int(math.ceil(float(train_data.samples) / batch_size))
    return num_batches, train_data


def reconstruct(encoder, decoder, input_fname, img_dim=(64, 64)):
    img = cv2.resize(cv2.imread(input_fname), img_dim)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (1. / 255) * img
    
    enc_vec = encoder.predict(np.array([img]))
    re_img = decoder.predict(enc_vec)

    re_img = re_img.reshape(re_img.shape[1:])
    re_img = (255 * re_img).astype(np.uint8)
    return re_img


def sample_grid_2d(decoder,
                   n_x=10, x_min=-10, x_max=10,
                   n_y=10, y_min=-10, y_max=10,
                   img_dim=(64, 64, 3)):
    grid_x = np.linspace(x_min, x_max, n_x)
    grid_y = np.linspace(y_min, y_max, n_y)

    figure = np.zeros([img_dim[0] * n_x,
                       img_dim[1] * n_y,
                       img_dim[2]])
    for i, xi in enumerate(grid_x):
        for j, yj in enumerate(grid_y):
            img = decoder.predict(np.array([[xi, yj]]))
            figure[i * img_dim[0]: (i + 1) * img_dim[0],
                   j * img_dim[1]: (j + 1) * img_dim[1],
                   :] = img.reshape(*img_dim)

    return figure


class GenSampler(object):
    def __init__(self, num_samples=32, latent_dim=128):
        self.noise = np.random.normal(
            size=[num_samples, latent_dim]).astype('float32')

    def sample(self, generator):
        sample_imgs = generator.predict_on_batch(self.noise)
        return sample_imgs

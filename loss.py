"""
Wasserstein loss
"""
import keras.backend as backend

import numpy as np

def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples, penalty_weight):
    grad = backend.gradients(y_pred, averaged_samples)[0]
    grad_sqr = backend.square(grad)
    grad_sum = backend.sum(grad_sqr,
                           axis=np.arange(1, len(grad_sqr.shape)))
    grad_norm = backend.sqrt(grad_sum)
    grad_penalty = penalty_weight * backend.square(1 - grad_norm)
    return backend.mean(grad_penalty)

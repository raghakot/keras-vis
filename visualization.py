from optimizer import Optimizer
from losses import ActivationMaximization
from utils import deprocess_image, get_image_indices

import numpy as np


class Saliency(object):
    """
    Computes the importance of pixels in input image for the filter output to be maximized.
    """
    def visualize(self, model, layer, filter_indices, seed_img):
        losses = [
            (ActivationMaximization(layer, filter_indices), 1)
        ]
        opt = Optimizer(model, losses)
        seed_img, grads = opt.minimize(max_iter=1, verbose=True, seed_img=seed_img)

        s, c, w, h = get_image_indices()
        grads = np.max(np.abs(grads), axis=c, keepdims=True)
        return deprocess_image(grads[0])

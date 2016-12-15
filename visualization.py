from optimizer import Optimizer
from losses import ActivationMaximization
from utils import deprocess_image, get_image_indices

import numpy as np
import cv2


class Saliency(object):
    """
    Generates a heatmap over input pixels indicating the pixels that contributed the most towards
    maximizing `filter_indices` output in the given `layer`
    """
    def visualize(self, input_layer, layer, filter_indices, seed_img):
        """
        :param input_layer: 4D Keras image input layer (including #samples)
        :param layer: The keras layer to visualize.
        :param filter_indices: filter indices within the layer to be maximized.
            For `Dense` layers, `filter_idx` is interpreted as output index.

            If you are optimizing final Dense layer to maximize class output, you tend to get
            better results with 'linear' activation as opposed to 'softmax'. This is because 'softmax'
            output can be maximized by minimizing scores for other classes.
        :param seed_img: The input image for which activation map needs to be visualized.
        :return: The input image overlayed with heatmap, Red values indicate higher pixel importance.
        """

        losses = [
            (ActivationMaximization(layer, filter_indices), 1)
        ]
        opt = Optimizer(input_layer, losses)
        _, grads = opt.minimize(max_iter=1, verbose=True, jitter=0, seed_img=seed_img)

        s, c, w, h = get_image_indices()
        grads = np.max(np.abs(grads), axis=c, keepdims=True)

        # Smoothen activation map
        grads = deprocess_image(grads[0])
        grads /= np.max(grads)

        # Convert to heatmap and zero out low probabilities for a cleaner output.
        heatmap = cv2.applyColorMap(cv2.GaussianBlur(grads * 255, (3, 3), 0), cv2.COLORMAP_JET)
        heatmap[np.where(grads <= 0.2)] = 0
        return cv2.addWeighted(seed_img, 1, heatmap, 0.5, 0)

from optimizer import Optimizer
from losses import ActivationMaximization
from regularizers import TotalVariation, LPNorm
from keras import backend as K

import utils
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

        s, c, w, h = utils.get_image_indices()
        grads = np.max(np.abs(grads), axis=c, keepdims=True)

        # Smoothen activation map
        grads = utils.deprocess_image(grads[0])
        grads /= np.max(grads)

        # Convert to heatmap and zero out low probabilities for a cleaner output.
        heatmap = cv2.applyColorMap(cv2.GaussianBlur(grads * 255, (3, 3), 0), cv2.COLORMAP_JET)
        heatmap[np.where(grads <= 0.2)] = 0
        return cv2.addWeighted(seed_img, 1, heatmap, 0.5, 0)


class LayerActivation(object):
    """
    Generates stitched input image(s) over all filters in a layer that maximize the filter output activation.
    """

    def _get_num_filters(self, layer):
        """
        :return: Total number of filters within this layer
        """
        # For all other layers it is 4
        isDense = K.ndim(layer.output) == 2

        if isDense:
            return layer.output.shape[1]
        else:
            if K.image_dim_ordering() == 'th':
                return layer.output.shape[1]
            else:
                return layer.output.shape[3]

    def visualize(self, input_layer, layer, filter_indices=None,
                  act_max_weight=1, lp_norm_weight=10, tv_weight=10,
                  show_filter_idx_text=True, idx_label_map=None, cols=10):
        """
        :param input_layer: 4D Keras image input layer (including #samples)
        :param layer: The keras layer to visualize.
        :param filter_indices: filter indices within the layer that needs to be visualized.
            If None, all filters are visualized.
        :param act_max_weight: The weight param for `ActivationMaximization` loss. Not used if 0 or None.
        :param lp_norm_weight: The weight param for `LPNorm` regularization loss. Not used if 0 or None.
        :param tv_weight: The weight param for `TotalVariation` regularization loss. Not used if 0 or None.
        :param show_filter_idx_text: Adds filter_idx text to the image if set to True.
        :param idx_label_map: Map of filter_idx to text label. If not None, this map is used to
            translate filter_idx to text value when show_filter_idx_text = True.
        :param cols: The number of cols to use, when generating stitched images.
        :return: Stitched image output visualizing input images that maximize the filter output(s).
        """
        if filter_indices is None:
            filter_indices = np.arange(self._get_num_filters(layer))

        imgs = []
        for i, idx in enumerate(filter_indices):
            losses = [
                (ActivationMaximization(layer, [idx]), act_max_weight or 0),
                (LPNorm(), lp_norm_weight or 0),
                (TotalVariation(), tv_weight or 0)
            ]
            opt = Optimizer(input_layer, losses)
            print('Working on filter {}/{}'.format(i + 1, len(filter_indices)))
            img, g = opt.minimize(verbose=False)

            # Add filter text to image if applicable.
            if show_filter_idx_text:
                label = None
                if idx_label_map:
                    label = idx_label_map.get(idx)
                if label is None:
                    label = "Filter {}".format(idx)
                cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            imgs.append(img)

        return utils.stitch_images(imgs, cols=cols)

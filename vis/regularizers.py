from __future__ import division

import numpy as np
from keras import backend as K

from losses import Loss
from utils import utils


def normalize(img, value):
    """
    Normalizes the value with respect to image dimensions. This makes regularizer weight factor more or less
    uniform across various input image dimensions.

    Args:
        img: 4D tensor with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
                `(samples, rows, cols, channels)` if dim_ordering='tf'.
        value: The function to normalize

    Returns:
        The normalized expression.
    """
    samples, c, w, h = utils.get_img_shape(img)
    return value / (c * w * h)


class TotalVariation(Loss):

    def __init__(self, beta=2.0):
        """Total variation regularizer encourages blobbier and coherent image structures, akin to natural images.
        See `section 3.2.2` in
        [Visualizing deep convolutional neural networks using natural pre-images](https://arxiv.org/pdf/1512.02017v3.pdf)
        for details.

        Args:
            beta: Smaller values of beta give sharper but 'spikier' images.
            Values \(\in [1.5, 2.0]\) are recommended as a reasonable compromise.
        """
        super(TotalVariation, self).__init__()
        self.name = "TV Loss"
        self.beta = beta

    def build_loss(self, img):
        r"""Implements the function
        $$TV^{\beta}(x) = \sum_{whc} \left ( \left ( x(h, w+1, c) - x(h, w, c) \right )^{2} +
        \left ( x(h+1, w, c) - x(h, w, c) \right )^{2} \right )^{\frac{\beta}{2}}$$
        """
        assert 4 == K.ndim(img)
        a = K.square(img[utils.slicer[:, :, 1:, :-1]] - img[utils.slicer[:, :, :-1, :-1]])
        b = K.square(img[utils.slicer[:, :, :-1, 1:]] - img[utils.slicer[:, :, :-1, :-1]])
        tv = K.sum(K.pow(a + b, self.beta/2.))
        return normalize(img, tv)


class LPNorm(Loss):

    def __init__(self, p=6.):
        """
        Builds a L-p norm function. This regularizer encourages the intensity of pixels to stay bounded.
            i.e., prevents pixels from taking on very large values.

        Args:
            p: The pth norm to use. If p = float('inf'), infinity-norm will be used.
        """
        super(LPNorm, self).__init__()
        if p < 1:
            raise ValueError('p value should range between [1, inf)')
        self.name = "L-{} Norm Loss".format(p)
        self.p = p

    def build_loss(self, img):
        # Infinity norm
        if np.isinf(self.p):
            value = K.max(img)
        else:
            value = K.pow(K.sum(K.pow(K.abs(img), self.p)), 1. / self.p)

        return normalize(img, value)

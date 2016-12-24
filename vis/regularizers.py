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
    return value / np.prod(K.int_shape(img)[1:])


class TotalVariation(Loss):

    def __init__(self, img_input, beta=2.0):
        """Total variation regularizer encourages blobbier and coherent image structures, akin to natural images.
        See `section 3.2.2` in
        [Visualizing deep convolutional neural networks using natural pre-images](https://arxiv.org/pdf/1512.02017v3.pdf)
        for details.

        Args:
            img_input: 4D image input tensor to the model of shape: `(samples, channels, rows, cols)`
                if dim_ordering='th' or `(samples, rows, cols, channels)` if dim_ordering='tf'.
            beta: Smaller values of beta give sharper but 'spikier' images.
                Values \(\in [1.5, 2.0]\) are recommended as a reasonable compromise.
        """
        super(TotalVariation, self).__init__()
        self.name = "TV Loss"
        self.beta = beta
        self.img = img_input

    def build_loss(self):
        r"""Implements the function
        $$TV^{\beta}(x) = \sum_{whc} \left ( \left ( x(h, w+1, c) - x(h, w, c) \right )^{2} +
        \left ( x(h+1, w, c) - x(h, w, c) \right )^{2} \right )^{\frac{\beta}{2}}$$
        """
        assert 4 == K.ndim(self.img)
        a = K.square(self.img[utils.slicer[:, :, 1:, :-1]] - self.img[utils.slicer[:, :, :-1, :-1]])
        b = K.square(self.img[utils.slicer[:, :, :-1, 1:]] - self.img[utils.slicer[:, :, :-1, :-1]])
        tv = K.sum(K.pow(a + b, self.beta/2.))
        return normalize(self.img, tv)


class LPNorm(Loss):

    def __init__(self, img_input, p=6.):
        """
        Builds a L-p norm function. This regularizer encourages the intensity of pixels to stay bounded.
            i.e., prevents pixels from taking on very large values.

        Args:
            img_input: 4D image input tensor to the model of shape: `(samples, channels, rows, cols)`
                if dim_ordering='th' or `(samples, rows, cols, channels)` if dim_ordering='tf'.
            p: The pth norm to use. If p = float('inf'), infinity-norm will be used.
        """
        super(LPNorm, self).__init__()
        if p < 1:
            raise ValueError('p value should range between [1, inf)')
        self.name = "L-{} Norm Loss".format(p)
        self.p = p
        self.img = img_input

    def build_loss(self):
        # Infinity norm
        if np.isinf(self.p):
            value = K.max(self.img)
        else:
            value = K.pow(K.sum(K.pow(K.abs(self.img), self.p)), 1. / self.p)

        return normalize(self.img, value)

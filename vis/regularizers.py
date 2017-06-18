from __future__ import absolute_import
from __future__ import division

import numpy as np
from keras import backend as K

from .losses import Loss
from .utils import utils


def normalize(img, tensor):
    """
    Normalizes the tensor with respect to image dimensions. This makes regularizer weight factor more or less
    uniform across various input image dimensions.

    Args:
        img: An tensor of shape: `(samples, channels, image_dims...)` if data_format='channels_first' or
            `(samples, image_dims..., channels)` if data_format='channels_last'.
        tensor: The tensor to normalize

    Returns:
        The normalized tensor.
    """
    image_dims = utils.get_img_shape(img)[1:]
    return tensor / np.prod(image_dims)


class TotalVariation(Loss):

    def __init__(self, img_input, beta=2.):
        """Total variation regularizer encourages blobbier and coherent image structures, akin to natural images.
        See `section 3.2.2` in
        [Visualizing deep convolutional neural networks using natural pre-images](https://arxiv.org/pdf/1512.02017v3.pdf)
        for details.

        Args:
            img_input: An image tensor of shape: `(samples, channels, image_dims...)` if data_format='channels_first' or
                `(samples, image_dims..., channels)` if data_format='channels_last'.
            beta: Smaller values of beta give sharper but 'spikier' images.
                Values \(\in [1.5, 3.0]\) are recommended as a reasonable compromise. (Default value = 2.)
        """
        super(TotalVariation, self).__init__()
        self.name = "TV({}) Loss".format(beta)
        self.img = img_input
        self.beta = beta

    def build_loss(self):
        r"""Implements the N-dim version of function
        $$TV^{\beta}(x) = \sum_{whc} \left ( \left ( x(h, w+1, c) - x(h, w, c) \right )^{2} +
        \left ( x(h+1, w, c) - x(h, w, c) \right )^{2} \right )^{\frac{\beta}{2}}$$
        to return total variation for all images in the batch.
        """
        image_dims = K.ndim(self.img) - 2

        # Constructing slice [1:] + [:-1] * (image_dims - 1) and [:-1] * (image_dims)
        start_slice = [slice(1, None, None)] + [slice(None, -1, None) for _ in range(image_dims - 1)]
        end_slice = [slice(None, -1, None) for _ in range(image_dims)]
        samples_channels_slice = [slice(None, None, None), slice(None, None, None)]

        # Compute pixel diffs by rolling slices to the right per image dim.
        tv = None
        for i in range(image_dims):
            ss = tuple(samples_channels_slice + start_slice)
            es = tuple(samples_channels_slice + end_slice)
            diff_square = K.square(self.img[utils.slicer[ss]] - self.img[utils.slicer[es]])
            tv = diff_square if tv is None else tv + diff_square

            # Roll over to next image dim
            start_slice = np.roll(start_slice, 1).tolist()
            end_slice = np.roll(end_slice, 1).tolist()

        tv = K.sum(K.pow(tv, self.beta / 2.))
        return normalize(self.img, tv)


class LPNorm(Loss):

    def __init__(self, img_input, p=6.):
        """
        Builds a L-p norm function. This regularizer encourages the intensity of pixels to stay bounded.
            i.e., prevents pixels from taking on very large values.

        Args:
            img_input: 4D image input tensor to the model of shape: `(samples, channels, rows, cols)`
                if data_format='channels_first' or `(samples, rows, cols, channels)` if data_format='channels_last'.
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

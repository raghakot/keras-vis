from __future__ import division

import numpy as np
from keras import backend as K
from utils import utils


class Regularizer(object):
    """
    Abstract class for defining an image regularization prior.
    """
    def __init__(self):
        self.name = "Unnamed Regularization Loss"

    def build_loss(self, img):
        """
        Define the loss that needs to be minimized.
        :param img: The image of shape (1, C, W, H)
        :return: The loss expression.
        """
        raise NotImplementedError()


class TotalVariation(Regularizer):
    """
    This regularizer encourages piecewise blobby structures akin to natural images.
    See section 3.2.2 in https://arxiv.org/pdf/1512.02017v3.pdf for details.
    """
    def __init__(self, beta=2.0):
        super(TotalVariation, self).__init__()
        self.name = "Total Variation Loss"
        self.beta = beta

    def build_loss(self, img):
        assert 4 == K.ndim(img)
        a = K.square(img[utils.slicer[:, :, 1:, :-1]] - img[utils.slicer[:, :, :-1, :-1]])
        b = K.square(img[utils.slicer[:, :, :-1, 1:]] - img[utils.slicer[:, :, :-1, :-1]])
        tv = K.sum(K.pow(a + b, self.beta/2.))

        samples, c, w, h = utils.get_img_shape(img)
        norm_tv = tv / (c * w * h)
        return norm_tv


class LPNorm(Regularizer):
    """
    This regularizer encourages the intensity of pixels to stay bounded.
    i.e., prevents pixels from taking on very large values.
    """
    def __init__(self, p=2.):
        """
        Builds an L-p Norm function.
        :param p: The pth norm to use. if p = float('inf'), infinity-norm will be used.
        """
        super(LPNorm, self).__init__()
        if p < 1:
            raise ValueError('p value should range between [1, inf)')
        self.name = "L-{} Norm Loss".format(p)
        self.p = p

    def build_loss(self, img):
        samples, c, w, h = utils.get_img_shape(img)

        # Infinity norm
        if np.isinf(self.p):
            value = K.max(img)
        else:
            value = K.pow(K.sum(K.pow(K.abs(img), self.p)), 1. / self.p)

        normed = value / (c * w * h)
        return normed

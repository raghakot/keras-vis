from utils import slicer, get_img_shape, get_image_indices
from keras import backend as K


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
        a = K.square(img[slicer[:, :, 1:, :-1]] - img[slicer[:, :, :-1, :-1]])
        b = K.square(img[slicer[:, :, :-1, 1:]] - img[slicer[:, :, :-1, :-1]])
        tv = K.sum(K.pow(a + b, self.beta/2.))

        samples, c, w, h = get_img_shape(img)
        norm_tv = tv / (w * h)
        return norm_tv


class BoundedRange(Regularizer):
    """
    This regularizer encourages the intensity of pixels to stay bounded.
    See section 3.2.1 in https://arxiv.org/pdf/1512.02017v3.pdf for details.
    """
    def __init__(self, alpha=6):
        super(BoundedRange, self).__init__()
        self.name = "Bounded Range Loss"
        self.alpha = alpha

    def build_loss(self, img):
        samples_idx, channel_idx, width_idx, height_idx = get_image_indices()
        samples, c, w, h = get_img_shape(img)

        value = K.sum(K.pow(K.square(K.sum(img, axis=channel_idx)), self.alpha / 2.))
        # 80 comes from section 3.3 in https://arxiv.org/pdf/1512.02017v3.pdf
        normed = value / (w * h * 80)
        return normed

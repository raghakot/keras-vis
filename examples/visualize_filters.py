from keras.applications import vgg16
import cv2

from optimizer import Optimizer
from losses import ActivationMaximization
from regularizers import TotalVariation


def main():
    """
    Visualize filters via activation maximization backprop.
    """

    # build the VGG16 network with ImageNet weights
    model = vgg16.VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # the name of the layer we want to visualize
    # (see model definition at keras/applications/vgg16.py)
    layer_name = 'block2_conv2'

    for filter_idx in range(10):
        regularizers = [(TotalVariation(), 0.01)]
        losses = [(ActivationMaximization(), 1)]
        opt = Optimizer(model, regularizers, losses, max_iter=200,
                        layer_name=layer_name, filter_idx=filter_idx)
        img = opt.minimize()
        cv2.imshow('filter_i'.format(filter_idx), img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()

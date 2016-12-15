import cv2
import utils
import numpy as np

from visualization import Saliency
from vggnet import VGG16


def main():
    """
    Visualize filters via activation maximization backprop.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    seed_img = utils.load_img('../resources/ouzel_1.jpg', target_size=(224, 224))
    saliency = Saliency()
    grads_image = saliency.visualize(model, layer_dict[layer_name], [20],
                                     np.array([seed_img], dtype=np.float32))
    cv2.imshow('Importance map', utils.stitch_images([seed_img, grads_image], cols=1))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

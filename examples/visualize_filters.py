import cv2
import utils

from losses import ActivationMaximization
from optimizer import Optimizer
from regularizers import TotalVariation, LPNorm
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

    imgs = []
    filter_indices = []
    for filter_idx in range(20, 21, 1):
        filter_indices.append(filter_idx)
        for i in range(3):
            losses = [
                (ActivationMaximization(layer_dict[layer_name], [filter_idx]), 1),
                (LPNorm(p=6.), 100),
                (TotalVariation(), 10)
            ]
            opt = Optimizer(model, losses)
            print('Working on filter {}_{}'.format(filter_idx, i))
            img = opt.minimize(max_iter=500, verbose=True)
            imgs.append(img)

    labels = ', '.join([utils.get_imagenet_label(idx) for idx in filter_indices])
    cv2.imshow('{}'.format(labels), utils.stitch_images(imgs))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

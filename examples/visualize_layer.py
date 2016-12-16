import cv2
import numpy as np

from utils import utils
from utils.vggnet import VGG16
from visualization import LayerActivation


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

    # Visualize couple random categories from imagenet.
    act_vis = LayerActivation()
    indices = np.random.permutation(1000)[:20]
    idx_label_map = dict((idx, utils.get_imagenet_label(idx)) for idx in indices)

    vis_img = act_vis.visualize(model.input, layer_dict[layer_name],
                                filter_indices=indices, idx_label_map=idx_label_map)
    cv2.imwrite('filters.jpg', vis_img)

if __name__ == '__main__':
    main()

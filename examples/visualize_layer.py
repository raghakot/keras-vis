import cv2
import numpy as np

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation


def visualize_random():
    """Example to show how to visualize multiple filters via activation maximization
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    # Visualize couple random categories from imagenet.
    indices = np.random.permutation(1000)[:15]
    idx_label_map = dict((idx, utils.get_imagenet_label(idx)) for idx in indices)

    vis_img = visualize_activation(model.input, layer_dict[layer_name], max_iter=500,
                                   filter_indices=indices, idx_label_map=idx_label_map)
    cv2.imshow('Random imagenet output categories', vis_img)
    cv2.waitKey(0)


def visualize_multiple_same_filter():
    """Example to show how to visualize same filter multiple times via different runs.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    # 20 is the imagenet category for 'ouzel'
    indices = [20, 20, 20]
    idx_label_map = dict((idx, utils.get_imagenet_label(idx)) for idx in indices)

    vis_img = visualize_activation(model.input, layer_dict[layer_name], max_iter=500,
                                   filter_indices=indices, idx_label_map=idx_label_map)
    cv2.imshow('Multiple runs of ouzel', vis_img)
    cv2.waitKey(0)


def visualize_multiple_categories():
    """Example to show how to visualize images that activate multiple categories
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    # Visualize [20] (ouzel) and [20, 71] (An ouzel-scorpion :D)
    indices = [20, [20, 71]]
    idx_label_map = dict((idx, utils.get_imagenet_label(idx)) for idx in [20, 71])

    vis_img = visualize_activation(model.input, layer_dict[layer_name], max_iter=500,
                                   filter_indices=indices, idx_label_map=idx_label_map)
    cv2.imshow('Multiple category visualization', vis_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    print('Visualizing random imagenet output categories')
    visualize_random()

    print('Visualizing same filter over multiple runs')
    visualize_multiple_same_filter()

    print('Visualizing multiple categories')
    visualize_multiple_categories()

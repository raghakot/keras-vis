import cv2

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam


def generate_saliceny_map():
    """Generates a heatmap indicating the pixels that contributed the most towards
    maximizing the filter output.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['../resources/ouzel.jpg', '../resources/ouzel_1.jpg']:
        seed_img = utils.load_img(path, target_size=(224, 224))
        # 20 is the imagenet category for 'ouzel'
        heatmap = visualize_saliency(model, layer_idx, [20], seed_img)
        cv2.imshow('Importance map', heatmap)
        cv2.waitKey(0)


def generate_cam():
    """Generates a heatmap via grad-CAM method.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

    for path in ['../resources/ouzel.jpg', '../resources/ouzel_1.jpg']:
        seed_img = utils.load_img(path, target_size=(224, 224))
        # 20 is the imagenet category for 'ouzel'
        heatmap = visualize_cam(model, layer_idx, [20], seed_img)
        cv2.imshow('Importance map', heatmap)
        cv2.waitKey(0)


if __name__ == '__main__':
    generate_cam()

import cv2

from utils import utils
from utils.vggnet import VGG16
from visualization import visualize_saliency


def main():
    """Generates a heatmap indicating the pixels that contributed the most towards
    maximizing the filter output.
    """
    # Build the VGG16 network with ImageNet weights
    model = VGG16(weights='imagenet', include_top=True)
    print('Model loaded.')

    # The name of the layer we want to visualize
    # (see model definition in vggnet.py)
    layer_name = 'predictions'
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    for path in ['../resources/ouzel.jpg', '../resources/ouzel_1.jpg']:
        seed_img = utils.load_img(path, target_size=(224, 224))
        # 20 is the imagenet category for 'ouzel'
        heatmap = visualize_saliency(model.input, layer_dict[layer_name], [20], seed_img)
        cv2.imshow('Importance map', heatmap)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()

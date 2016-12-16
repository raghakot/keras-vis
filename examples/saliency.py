import cv2

from utils import utils
from utils.vggnet import VGG16
from visualization import Saliency


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
    heatmap = saliency.visualize(model.input, layer_dict[layer_name], [20], seed_img)
    cv2.imshow('Importance map', heatmap)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

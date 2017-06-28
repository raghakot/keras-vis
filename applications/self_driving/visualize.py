import numpy as np
from matplotlib import pyplot as plt

from model import build_model, FRAME_W, FRAME_H
from keras.preprocessing.image import img_to_array

from vis.utils import utils
from vis.visualization import visualize_regression_cam, visualize_regression_saliency, overlay


def visualize_attention(model, img_path, target):
    """Visualize attention for self driving `model`.

    Args:
        model: The keras model.
        img_path: The image path to use as input.
        target: One of 'right', 'left', 'same' to indicate whether regression target should 'increase', 'decrease'
            or remain 'same'.
    """
    img = utils.load_img(img_path, target_size=(FRAME_H, FRAME_W))

    # Convert to BGR, create input with batch_size: 1.
    bgr_img = utils.bgr2rgb(img)
    img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
    pred = model.predict(img_input)[0][0]
    print('Predicted {}'.format(pred))

    if target == 'right':
        t = pred + 1
    elif target == 'left':
        t = pred - 1
    else:
        t = pred

    # Generate saliency visualization.
    saliency = visualize_regression_saliency(model, layer_idx=-1, output_indices=0, targets=t, seed_input=bgr_img)
    saliency = overlay(img, saliency, alpha=0.7)
    plt.figure()
    plt.axis('off')
    plt.title('Saliency to steer: {}'.format(target))
    plt.imshow(saliency)

    # Generate grad-CAM visualization.
    cam = visualize_regression_cam(model, layer_idx=-1, output_indices=0, targets=t,
                                   seed_input=bgr_img, penultimate_layer_idx=5)
    cam = overlay(img, cam, alpha=0.7)
    plt.figure()
    plt.axis('off')
    plt.title('grad-CAM to steer: {}'.format(target))
    plt.imshow(cam)

    plt.show()


if __name__ == '__main__':
    model = build_model()
    model.load_weights('weights.hdf5')

    visualize_attention(model, 'images/left.png', 'right')
    visualize_attention(model, 'images/left.png', 'left')

    # Note that same mode is currently experimental and needs more thinking.
    visualize_attention(model, 'images/left.png', 'same')

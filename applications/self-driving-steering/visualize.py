from model import *
import cv2
import scipy.misc
import numpy as np
import argparse


def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def grad_cam_loss(x, angle):
    if angle > 5.0 * scipy.pi / 180.0:
        return x
    elif angle < -5.0 * scipy.pi / 180.0:
        return -x
    else:
        return tf.inv(x) * np.sign(angle)


def grad_cam_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def visualize_grad_cam(input_model, original_img, layer_name="conv3_1"):
    img = np.float32(cv2.resize(original_img, (200, 66))) / 255.0

    angle = input_model.predict(np.array([img]))
    print("The predicted angle is", 180.0 * angle[0][0] / scipy.pi, "degrees")

    model = Sequential()
    model.add(input_model)

    target_layer = lambda x: grad_cam_loss(x, angle)
    model.add(Lambda(target_layer,
                     output_shape=grad_cam_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output = [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([[img]])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # ReLU:
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = cv2.resize(cam, tuple(original_img.shape[0:2][::-1]))

    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    cam = 1.0 * np.float32(cam) + np.float32(original_img)
    cam = cam / np.max(cam)
    return cam


def extract_hypercolumns(model, layer_indexes, image):
    layers = [model.layers[li].output for li in layer_indexes]
    get_feature = K.function([model.layers[0].input], layers)
    feature_maps = get_feature([[image]])
    hypercolumns = []
    for convmap in feature_maps:
        fmaps = [np.float32(convmap[0, :, :, i]) for i in range(convmap.shape[-1])]
        layer = []
        for fmap in fmaps:
            fmap = np.abs(fmap)
            norm = np.max(np.max(fmap, axis=0), axis=0)
            if norm > 0:
                fmap = fmap / norm

            upscaled = scipy.misc.imresize(fmap, size=(66, 200),
                                           mode="F", interp='bilinear')
            layer.append(upscaled)

        hypercolumns.append(np.mean(np.float32(layer), axis=0))

    return np.asarray(hypercolumns)


def visualize_hypercolumns(model, original_img):
    img = np.float32(cv2.resize(original_img, (200, 66))) / 255.0

    layers_extract = [9]

    hc = extract_hypercolumns(model, layers_extract, img)
    avg = np.product(hc, axis=0)
    avg = np.abs(avg)
    avg = avg / np.max(np.max(avg))

    heatmap = cv2.applyColorMap(np.uint8(255 * avg), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / np.max(np.max(heatmap))
    heatmap = cv2.resize(heatmap, original_img.shape[0:2][::-1])

    both = 255 * heatmap * 0.7 + original_img
    both = both / np.max(both)
    return both


def visualize_occlussion_map(model, original_img):
    imgs, windows = [], []
    img = cv2.resize(original_img, (200, 66))
    base_angle = model.predict(np.array([img]))

    for x in range(0, img.shape[1], 2):
        for y in range(0, img.shape[0], 2):
            windows.append((x, y, 15, 15))
            windows.append((x, y, 50, 50))

    for window in windows:
        x, y, w, h = window
        masked = img * 1
        masked[y: y + h, x: x + w] = 0
        imgs.append(masked)

    angles = model.predict(np.array(imgs))
    result = np.zeros(shape=img.shape[0:2], dtype=np.float32)
    for i, window in enumerate(windows):
        diff = np.abs(angles[i] - base_angle)
        x, y, w, h = window
        result[y: y + h, x: x + w] += diff
    mask = np.abs(result)
    mask = mask / np.max(np.max(mask))
    # mask[np.where(mask < np.percentile(mask, 60))] = 0
    mask = cv2.resize(mask, original_img.shape[0:2][::-1])

    result = original_img
    result[np.where(mask == 0)] = 0
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="Path of an image to run the network on")
    parser.add_argument("--output_path", type=str, default="out.jpg", help="Path of an image to run the network on")
    parser.add_argument("--model_path", type=str, help="Path of the trained model", default="weights.hdf5")
    parser.add_argument("--type", type=str, help="cam/hypercolumns/occlusion", default="cam")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    model = load_model(args.model_path)
    img = cv2.imread(args.image_path, 1)
    original_shape = img.shape
    img = np.float32(img)

    visualizations = {"cam": visualize_grad_cam, \
                      "hypercolumns": visualize_hypercolumns, \
                      "occlusion": visualize_occlussion_map}

    output = visualizations[args.type](model, img * 1)
    output = np.uint8(255 * output)
    output = cv2.resize(output, original_shape[0:2][::-1])

    cv2.imwrite(args.output_path, output)

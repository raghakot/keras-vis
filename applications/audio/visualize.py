import numpy as np
from keras.models import load_model
from vis.visualization import visualize_cam, visualize_saliency
from vis.utils import utils

from matplotlib import pyplot as plt
from keras import backend as K

K.set_image_data_format('channels_first')
seed_input = np.load('input.npy')
model = load_model('2017_06_22_00_36_17_model.h5')

pred_class = 4
penultimate_idx = utils.find_layer_idx(model, 'bidirectional_2')
heatmap_weak = visualize_saliency(model, utils.find_layer_idx(model, 'weak_out'),
                                  pred_class, seed_input, backprop_modifier='deconv')

heatmap_strong = visualize_saliency(model, utils.find_layer_idx(model, 'strong_out'),
                                    pred_class, seed_input, backprop_modifier='deconv')

plt.figure()
plt.subplot(311)
plt.imshow(np.transpose(heatmap_strong, (1, 0, 2)), aspect='auto', origin='lower')

plt.subplot(312)
plt.imshow(np.transpose(heatmap_weak, (1, 0, 2)), aspect='auto', origin='lower')

plt.subplot(313)
plt.imshow(seed_input[0].T, aspect='auto', origin='lower')
plt.show()

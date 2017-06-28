## Overview
Each conv layer has several learned 'template matching' filters that maximize their output when a similar template 
pattern is found in the input image. This makes the first conv net layer highly interpretable by simply visualizing 
their weights as it is operating over raw pixels.
 
Subsequent conv filters operate over the outputs of previous conv filters (which indicate the presence or absence of 
some templates), so visualizing them directly is not very interpretable.

One way of interpreting them is to generate an input image that maximizes the filter output. With keras-vis, setting
this up is easy. Lets visualize the second conv layer of vggnet (named as 'block1_conv2').

```python
import numpy as np
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_class_activation, get_num_filters

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'block1_conv2'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
vis_images = []
for idx in filters:
    img = visualize_class_activation(model, layer_idx, filter_indices=idx) 
    img = utils.draw_text(img, str(idx))
    vis_images.append(img)

# Generate stitched image palette with 8 cols.
stitched = utils.stitch_images(vis_images, cols=8)    
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()

```

This generates the following stitched image representing input image(s) that maximize the filter_idx output.
They mostly seem to match for specific color and directional patterns.

![block1_conv2 filters](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/conv_vis/block1_conv2_filters.jpg?raw=true "conv_1 filters")

Lets visualize the remaining conv filters (first few) by iterating over different `layer_name` values.

## block2_conv2: random sample of the 128 filters

![block2_conv2 filters](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/conv_vis/block2_conv2_filters.jpg?raw=true "conv_2 filters")

## block3_conv3: random sample of the 256 filters

![block3_conv3 filters](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/conv_vis/block3_conv3_filters.jpg?raw=true "conv_3 filters")

## block3_conv4: random sample of the 512 filters

![block4_conv3 filters](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/conv_vis/block4_conv3_filters.jpg?raw=true "conv_4 filters")

## block3_conv5: random sample of the 512 filters

![block5_conv3 filters](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/conv_vis/block5_conv3_filters.jpg?raw=true "conv_5 filters")

Some of the 'block5_conv3' filters failed to converge. This is because regularization losses (total variation and 
LP norm) are overtaking activation maximization loss (set `verbose=True` to observe). 

Whenever activation maximization fails to converge, total variation regularization is the typical culprit. 
It is easier to minimize total variation from a random image (just have to create blobbier color structures), 
and this sets the input image in a bad local minima that makes it difficult to optimize for activation maximization. 
We can turn off total variation by setting `tv_weight=0`. This generates most of the previously unconverged filters.

![block5_conv3 filters_no_tv](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/conv_vis/block5_conv3_filters_no_tv.jpg?raw=true "conv_5 filters_no_tv")

By this layer, we can clearly notice templates for complex patterns such as flower buds / corals 
(filters 67, 84 respectively). Notice that images are not as coherent due to lack of total variation loss.

A good strategy in these situations might be to seed the optimization with image output generated via tv_weight=0
and add the tv_weight back. Lets specifically look at filter 67.

```python
layer_name = 'block5_conv3'

no_tv_seed_img = visualize_class_activation(model, layer_idx, filter_indices=[67],
                                      tv_weight=0, verbose=True)
post_tv_img = visualize_class_activation(model, layer_idx, filter_indices=[67],
                                   tv_weight=1, seed_img=no_tv_seed_img, verbose=True, max_iter=300)
plt.axis('off')
plt.imshow(post_tv_img)
plt.title(layer_name)
plt.show()

```
As expected, this generates a blobbier and smoother image:

![filter_67](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/conv_vis/filter_67.png?raw=true "filter_75")

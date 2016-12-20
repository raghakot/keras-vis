# Keras Visualization Toolkit

keras-vis is a high-level toolkit for visualizing input images via guided backprop. 
There are several repositories out there to visualize: 

- Activation maximization
- Saliency maps
- Caricaturization (deep dream)
- Texture style transfer
- Neural style transfer

This toolkit generalizes all of the above and image backprop problems in general as energy minimization problem.
Compatible with both theano and tensorflow backends. 

## Getting Started
In image backprop problems, the goal is to generate an input image that minimizes some loss function.

Various useful loss functions are defined in [losses.py](https://github.com/raghakot/keras-vis/blob/master/losses.py).
A custom loss function can be defined by implementing [Loss](https://github.com/raghakot/keras-vis/blob/master/losses.py#L5)
class.

In order to generate natural looking images, image search space is constrained using regularization penalties. 
Some common regularizers are defined in [regularizers.py](https://github.com/raghakot/keras-vis/blob/master/regularizers.py).
Like loss functions, custom regularizer can be defined by implementing 
[Loss](https://github.com/raghakot/keras-vis/blob/master/losses.py#L5) class.

Setting up an image backprop problem is easy.

1. Define weighted loss function
```python
from losses import ActivationMaximization
from regularizers import TotalVariation, LPNorm

filter_indices = [1, 2, 3]

# Tuple consists of (loss_function, weight)
# Add regularizers as needed.
losses = [
    (ActivationMaximization(keras_layer, filter_indices), 1),
    (LPNorm(), 10),
    (TotalVariation(), 10)
]
```

2. Configure optimizer to minimize weighted loss
```python
from optimizer import Optimizer

optimizer = Optimizer(img_input_layer, losses)
opt_img, grads = optimizer.minimize()
```

## Quick start
See examples for various visualizations in [examples/](https://github.com/raghakot/keras-vis/tree/master/examples) folder.

## Visualizations
Neural nets are black boxes. How can we be sure that they are learning the right thing? If the neural net generates a
wrong prediction, how could we diagnose the issue? In the recent years, several approaches for understanding and 
visualizing Convolutional Networks have been developed in the literature.

### Conv filter visualization
Each conv layer has several learned 'template matching' filters that maximize their output when a similar template 
pattern is found in the input image. This makes the first conv net layer highly interpretable by simply visualizing 
their weights as it is operating over raw pixels.
 
Subsequent conv filters operate over the outputs of previous conv filters (which indicate the presence or absence of 
some templates), so visualizing them directly is not very interpretable.

One way of interpreting them is to generate an input image that maximizes the filter output. With keras-vis, setting
this up is easy. Lets visualize the second conv layer of vggnet (named as 'block1_conv2').

```python
import cv2
from utils.vggnet import VGG16
from visualization import visualize_activation

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'block1_conv2'
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

vis_img = visualize_activation(model.input, layer_dict[layer_name])
cv2.imshow(layer_name, vis_img)
cv2.waitKey(0)
```

This generates the following stitched image representing input image(s) that maximize the filter_idx output.
They mostly seem to match for specific color and directional patterns.

![block1_conv2 filters](images/block1_conv2_filters.jpg?raw=true "conv_1 filters")

Lets visualize the remaining conv filters (first few) by iterating over different `layer_name` values.

####block2_conv2: random sample of the 128 filters

![block2_conv2 filters](images/block2_conv2_filters.jpg?raw=true "conv_2 filters")

####block3_conv3: random sample of the 256 filters

![block3_conv3 filters](images/block3_conv3_filters.jpg?raw=true "conv_3 filters")

####block3_conv4: random sample of the 512 filters

![block4_conv3 filters](images/block4_conv3_filters.jpg?raw=true "conv_4 filters")

####block3_conv5: random sample of the 512 filters

![block5_conv3 filters](images/block5_conv3_filters.jpg?raw=true "conv_5 filters")

Some of the 'block5_conv3' filters failed to converge. This is because regularization losses (total variation and 
LP norm) are overtaking activation maximization loss (set `verbose=True` to observe). 

Whenever activation maximization fails to converge, total variation regularization is the typical culprit. 
It is easier to minimize total variation from a random image (just have to create blobbier color structures), 
and this sets the input image in a bad local minima that makes it difficult to optimize for activation maximization. 
We can turn off total variation by setting `tv_weight=0`. This generates most of the previously unconverged filters.

![block5_conv3 filters_no_tv](images/block5_conv3_filters_no_tv.jpg?raw=true "conv_5 filters_no_tv")

By this layer, we can clearly notice templates for complex patterns such as flower buds / corals 
(filters 67, 84 respectively). Notice that images are not as coherent due to lack of total variation loss.

A good strategy in these situations might be to seed the optimization with image output generated via tv_weight=0
and add the tv_weight back. Lets specifically look at filter 67.

```python
layer_name = 'block5_conv3'

no_tv_seed_img = visualize_activation(model.input, layer_dict[layer_name], filter_indices=[67],
                                      tv_weight=0, verbose=True)
post_tv_img = visualize_activation(model.input, layer_dict[layer_name], filter_indices=[67],
                                   tv_weight=1, seed_img=no_tv_seed_img, verbose=True, max_iter=300)
cv2.imshow(layer_name, post_tv_img)
cv2.waitKey(0)
```
As expected, this generates a blobbier and smoother image:

![filter_67](images/filter_67.png?raw=true "filter_75")

###Dense layer visualization
Given an input image, conv net can classify whether it is a cat, bird etc. How can we be sure that it is capturing 
the correct notion of what it means to be a bird? Suppose that all the training images of 'bird' class contains a tree
with leaves. How do we know that the conv net is indeed 'looking' at the bird as opposed to leaves and classifying it 
as a bird?

One way to peer into the black box is to ask the reverse question - Generate an input image that maximizes the final
`Dense` layer output corresponding to bird class. Lets try this for 'ouzel' (imagenet output category: 20)

```python
import cv2
from utils.vggnet import VGG16
from visualization import visualize_activation

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
layer_name = 'predictions'

# Generate three different images of the same output index.
img = visualize_activation(model.input, layer_dict[layer_name],
                           filter_indices=[20, 20, 20], max_iter=500)
cv2.imshow(layer_name, img)
cv2.waitKey(0)
```

and out comes this..

![ouzel_vis](images/ouzel_vis.png?raw=true "ouzel_vis")

Not only has the conv net captured what it means to be an ouzel, but it also seems to encode for different orientations 
and scales, a further proof of rotational and scale invariance. 

Lets do this for a few more random imagenet categories.

![random_imagenet](images/random_imagenet.png?raw=true "random_imagenet")

If you squint really hard, we can sort of see that most images are more or less accurate representations of the 
corresponding class.

You might notice that in most of these visualizations, the same pattern tends to repeat all over the image 
with different orientations and scales. Why is this the case? If you think about it, it is essentially the consequence
of activation maximization loss. Multiple copies of 'template pattern' all over the image should increase the output value.

If we want more natural looking images, we need a better 'natural image' regularizer that penalizes this sort of 
behavior. Instead of hand crafting the regularizer, we can use the negative of 'discriminator' output probability 
(since we want to maximize probability that the image is real) of a generative adversarial network (GAN). 

See this article for details about GANs in general: [Unsupervised Representation Learning with Deep Convolutional 
Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

*GAN regularizer is currently a work in progress. Check back in a few days.*

At this point, it might be fun to see the effect of total variation regularizer. The following images are generated with
`tv_weight=0`

![random_imagenet_no_tv](images/random_imagenet_no_tv.png?raw=true "random_imagenet_no_tv")

Total variation regularizer definitely helps in creating more natural looking images. I am excited to see what
a GAN discriminator could do.

### Saliency Maps

### More will be added soon (WIP...)

## Generating animated gif of optimization progress

It is possible to generate an animated gif of optimization progress. Below is an example for activation maximization
of 'ouzel' class (output_index: 20).

```python
from utils.vggnet import VGG16
from optimizer import Optimizer
from losses import ActivationMaximization
from regularizers import TotalVariation, LPNorm

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
output_class = [20]

losses = [
    (ActivationMaximization(layer_dict[layer_name], output_class), 1),
    (LPNorm(), 10),
    (TotalVariation(), 1)
]
opt = Optimizer(model.input, losses)

# Jitter is used as a regularizer to create crisper images, but it makes gif animation ugly.
opt.minimize(max_iter=500, verbose=True, jitter=0,
             progress_gif_path='opt_progress')
```

![opt_progress](images/opt_progress.gif?raw=true "Optimization progress")

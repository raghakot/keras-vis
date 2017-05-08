## Overview
How can we assess whether a network is over/under fitting or generalizing well? Given an input image, 
conv net can classify whether it is a cat, bird etc. How can we be sure that it is capturing the correct notion of 
what it means to be a bird?

One way to answer these questions is to pose the reverse question:
> Generate an input image that maximizes the final `Dense` layer output corresponding to bird class. 

Lets try this for 'ouzel' (imagenet output category: 20)

```python
from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation


# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Generate three different images of the same output index.
vis_images = []
for idx in [20, 20, 20]:
    img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=500)
    img = utils.draw_text(img, str(idx))
    vis_images.append(img)

stitched = utils.stitch_images(vis_images)    
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()

```

and out comes this..

![ouzel_vis](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/dense_vis/ouzel_vis.png?raw=true "ouzel_vis")

Not only has the conv net captured what it means to be an ouzel, but it also seems to encode for different orientations 
and scales, a further proof of rotational and scale invariance. 

Lets do this for a few more random imagenet categories.

![random_imagenet](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/dense_vis/random_imagenet.png?raw=true "random_imagenet")

If you squint really hard, we can sort of see that most images are more or less accurate representations of the 
corresponding class.

## Generating more natural looking images

You might notice that in most of these visualizations, the same pattern tends to repeat all over the image 
with different orientations and scales. Why is this the case? If you think about it, it is essentially the consequence
of using [ActivationMaximization](vis.losses#ActivationMaximization) loss. Multiple copies of 'template pattern' 
all over the image would certainly maximize the output value.

If we want more natural looking images, we need a better *natural image prior*. A natural image prior is something that
captures the degree of naturalness. By default [visualize_activation](vis.visualization#visualize_activation) uses:

* [TotalVariation](vis.regularizers#TotalVariation) regularizer to Prefer blobbier images. i.e., not bobby images emit higher loss values.
* [LPNorm](vis.regularizers#LPNorm) regularizer to limit the color range.

If we set `tv_weight=0`, i.e., turn off total variation regularization, the following is generated:

![random_imagenet_no_tv](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/dense_vis/random_imagenet_no_tv.png?raw=true "random_imagenet_no_tv")

Total variation regularizer definitely helps, but in order to get even more natural looking images, we need a better 
image prior that penalizes unnatural images. Instead of hand-crafting these losses, perhaps the best approach is to 
use a generative adversarial network (GAN) discriminator. A GAN discriminator is trained to emit probability that the 
input image is real/fake. To learn more about GANs in general, read: [Unsupervised Representation Learning with Deep Convolutional 
Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

*I am currently in the process of building a GAN regularizer. Stay tuned!*

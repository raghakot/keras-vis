###Dense layer visualization
How can we assess whether a network is over/under fitting or generalizing well? Given an input image, 
conv net can classify whether it is a cat, bird etc. How can we be sure that it is capturing the correct notion of 
what it means to be a bird? Suppose that all the training images of 'bird' class contains a tree
with leaves. How do we know that the conv net is indeed leveraging bird-related pixels as opposed to some 
other features such as the tree or leaves in the image.

One way to answer these questions is to pose the reverse question:
> Generate an input image that maximizes the final `Dense` layer output corresponding to bird class. 

Lets try this for 'ouzel' (imagenet output category: 20)

```python
import cv2
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
img = visualize_activation(model, layer_idx,
                           filter_indices=[20, 20, 20], max_iter=500)
cv2.imshow(layer_name, img)
cv2.waitKey(0)
```

and out comes this..

![ouzel_vis](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/dense_vis/ouzel_vis.png?raw=true "ouzel_vis")

Not only has the conv net captured what it means to be an ouzel, but it also seems to encode for different orientations 
and scales, a further proof of rotational and scale invariance. 

Lets do this for a few more random imagenet categories.

![random_imagenet](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/dense_vis/random_imagenet.png?raw=true "random_imagenet")

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

![random_imagenet_no_tv](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/dense_vis/random_imagenet_no_tv.png?raw=true "random_imagenet_no_tv")

Total variation regularizer definitely helps in creating more natural looking images. I am excited to see what
a GAN discriminator could do.
## What is a Class Activation Map?

Class activation maps or grad-CAM is another way of visualizing attention over input. Instead of using gradients with
respect to output (see [saliency](saliency)), grad-CAM uses penultimate (pre `Dense` layer) `Conv` layer output. The
intuition is to use the nearest `Conv` layer to utilize spatial information that gets completely lost in `Dense` layers.

In keras-vis, we use [grad-CAM](https://arxiv.org/pdf/1610.02391.pdf) as its considered more general than 
[Class Activation maps](http://cnnlocalization.csail.mit.edu/).

## Usage

There are two APIs exposed to visualize grad-CAM and are almost identical to [saliency usage](saliency#Usage).

1. [visualize_cam](../vis.visualization#visualize_cam): This is the general purpose API for visualizing
grad-CAM.
2. [visualize_cam_with_losses](../vis.visualization#visualize_cam_with_losses): This is intended for 
research use-cases where some custom weighted loss can be used.

The only notable addition is the `penultimate_layer_idx` parameter. This can be used to specify the pre-layer
whose output gradients are used. By default, keras-vis will search for the nearest layer with filters.

### Scenarios

See [saliency scenarios](saliency#scenarios). Everything is identical expect the added `penultimate_layer_idx` param.

## Gotchas

grad-CAM only works well if the penultimate layer is close to the layer being visualized. This also applies to `Conv` 
filter visualizations. You are better off using saliency of this is not the case with your model.

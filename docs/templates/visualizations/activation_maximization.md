## What is Activation Maximization?

In a CNN, each Conv layer has several learned *template matching* filters that maximize their output when a similar 
template pattern is found in the input image. First Conv layer is easy to interpret; simply visualize the weights as 
an image. To see what the Conv layer is doing, a simple option is to apply the filter over raw input pixels. 
Subsequent Conv filters operate over the outputs of previous Conv filters (which indicate the presence or absence 
of some templates), making them hard to interpret.

The idea behind activation maximization is simple in hindsight - Generate an input image that maximizes the filter 
output activations. i.e., we compute 

$$\frac{\partial ActivationMaximizationLoss}{\partial input}$$

and use that estimate to update the input. [ActivationMaximization](../vis.losses#activationmaximization) loss simply 
outputs small values for large filter activations (we are minimizing losses during gradient descent iterations). 
This allows us to understand what sort of input patterns activate a particular filter. For example, there could be 
an eye filter that activates for the presence of eye within the input image.

## Usage

There are two APIs exposed to perform activation maximization.

1. [visualize_activation](../vis.visualization#visualize_activation): This is the general purpose API for visualizing
activations.
2. [visualize_activation_with_losses](../vis.visualization#visualize_activation_with_losses): This is intended for 
research use-cases where some custom weighted losses can be minimized.

See [examples/](https://github.com/raghakot/keras-vis/tree/master/examples) for code examples.

### Scenarios

The API is very general purpose and can be used in a wide variety of scenarios. We will list the most common use-cases
below:

#### Categorical Output Dense layer visualization

How can we assess whether a network is over/under fitting or generalizing well? Given an input image, a CNN can 
classify whether it is a cat, bird etc. How can we be sure that it is capturing the correct notion of what it means 
to be a bird? 
  
One way to answer these questions is to pose the reverse question:
> Generate an input image that maximizes the final `Dense` layer output corresponding to bird class. 

This can be done by pointing `layer_idx` to final `Dense` layer, and setting `filter_indices` to the desired output 
category. 

- For multi-class classification, `filter_indices` can point to a single class. You could point also point it to 
multiple categories to see what a cat-fish might look like, as an example.
- For multi-label classifier, simply set the appropriate `filter_indices`.

#### Regression Output Dense layer visualization

Unlike class activation visualizations, for regression outputs, we could visualize input that 

- increases
- decreases

the regressed `filter_indices` output. For example, if you trained an apple counter model, increasing the regression
output should correspond to more apples showing up in the input image. Similarly one could decrease the current output.
This can be achieved by using `grad_modifier` option. As the name suggests, it is used to modify the gradient of losses
with respect to inputs. By default, `ActivationMaximization` loss is used to increase the output. By setting 
`grad_modifier='negate'` you can negate the gradients, thus causing output values to decrease. 
[gradient_modifiers](../vis.grad_modifiers) are very powerful and show up in other visualization APIs as well. 


#### Conv filter visualization

By pointing `layer_idx` to `Conv` layer, you can visualize what pattern activates a filter. This might help you discover
what a filter might be computing. Here, `filter_indices` refers to the index of the `Conv` filter within the layer.

### Advanced usage

[backprop_modifiers](../vis.backprop_modifiers) allow you to modify the backpropagation behavior. For examples, 
you could tweak backprop to only propagate positive gradients by using `backprop_modifier='relu'`. This parameter also 
accepts a function and can be used to implement your crazy research idea :)

## Tips and tricks

- If you get garbage visualization, try setting `verbose=True` to see various losses during gradient descent iterations.
By default, `visualize_activation` uses `TotalVariation` and `LpNorm` regularization to enforce natural image prior. It
is very likely that you would see `ActivationMaximization Loss` bounce back and forth as they are dominated by regularization 
loss weights. Try setting all weights to zero and gradually try increasing values of total variation weight.

- To get sharper looking images, use [Jitter](../vis.input_modifiers#jitter) input modifier.

- Regression models usually do not provide enough gradient information to generate meaningful input images. Try seeding
the input using `seed_input` and see if the modifications to the input make sense.

- Consider submitting a PR to add more tips and tricks that you found useful.

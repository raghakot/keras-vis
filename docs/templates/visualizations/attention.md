## Overview
Suppose that all the training images of 'bird' class contains a tree with leaves. How do we know that the conv net is 
indeed leveraging bird-related pixels as opposed to some other features such as the tree or leaves in the image. 

Attention maps are a family of methods that try to answer these questions by generating a heatmap over input 
image that most contributed towards maximizing the probability of an output class.

## Saliency maps
Saliency maps was first introduced in the paper: 
[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/pdf/1312.6034v2.pdf)

The idea is pretty simple. We compute the gradient of output category with respect to input image. This should tell us
how output category value changes with respect to a small change in input image pixels. All the positive values
in the gradients tell us that a small change to that pixel will increase the output value. 
Hence, visualizing these gradients, which are the same shape as the image should provide some intuition of attention.

keras-vis abstracts all of this under the hood with [visualize_class_saliency](../vis.visualization/#visualize_class_saliency). 
Lets try to visualize attention over images with: *tiger, penguin, dumbbell, speedboat, spider*. Note there is no guarantee
that these image urls haven't expired. Update them as needed.
 
```python
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_class_saliency, overlay

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
image_paths = [
    "http://www.tigerfdn.com/wp-content/uploads/2016/05/How-Much-Does-A-Tiger-Weigh.jpg",
    "http://www.slate.com/content/dam/slate/articles/health_and_science/wild_things/2013/10/131025_WILD_AdeliePenguin.jpg.CROP.promo-mediumlarge.jpg",
    "https://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
    "http://tampaspeedboatadventures.com/wp-content/uploads/2010/10/DSC07011.jpg",
    "http://ichef-1.bbci.co.uk/news/660/cpsprodpb/1C24/production/_85540270_85540265.jpg"
]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(path, target_size=(224, 224))
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_class_saliency(model, layer_idx, [pred_class], seed_img)

    # Overlay heatmap onto the image with alpha blend.
    heatmaps.append(overlay(seed_img, heatmap))

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.show()

```

This generates heatmaps overlayed on top of images. overlay can be turned off using `overlay=False`.
 
![saliency_map](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/attention_vis/saliency_map.png?raw=true "saliency map")

This mostly looks pretty accurate! Note that the heatmap looks pretty sparse as the `Dense` layers destroys a lot of spatial 
information. This visualization should look a lot better if we used 1 X 1 convolutions to 
[mimic](http://cs231n.github.io/convolutional-networks/#convert) the dense layer.

## Class activation maps
As you might expect, since the inception of saliency maps by Simonyan et al, various other techniques have been developed 
to improve upon these visualizations. One problem with saliency maps is that it is not class discriminative; i.e., there
is some overlap in heatmaps between, say the 'dog' and 'cat' class. Notable methods to solve this problem includes:  

* [Occulusion maps](https://arxiv.org/pdf/1311.2901v3.pdf)
* [Class Activation maps](http://cnnlocalization.csail.mit.edu/)

In keras-vis, we however adopt the [grad-CAM](https://arxiv.org/pdf/1610.02391v1.pdf) method as it solves the inefficiency
problem with occlusion maps and architectural constraint problem with CAM.

Generating grad-CAM visualization is simple, just replace `visualize_class_saliency` with 
[visualize_class_cam](../vis.visualization/#visualize_class_cam) in the above code. This generates the following:

![grad-cam](https://raw.githubusercontent.com/raghakot/keras-vis/master/images/attention_vis/grad-cam.png?raw=true "grad cam")

Compared to saliency, notice how this excludes the spider the in `spider_web` prediction. I personally feel that grad-CAM 
is more helpful in diagnosing issues with conv-nets, especially for Kaggle competitions. 

## Advanced use cases

Internally, `visualize_class_saliency` and `visualize_class_cam` use 
[visualize_saliency](../vis.visualization/#visualize_saliency) and [visualize_cam](../vis.visualization/#visualize_cam) 
respectively with [ActivationMaximization](vis.losses#ActivationMaximization) loss. These methods allow custom loss 
functions to be used. For example, if the output of your model is not a class but a regression output (for example, 
predicting the age), then a different loss function needs to be used. This is precisely what 
[visualize_regression_saliency](../vis.visualization/#visualize_regression_saliency) and 
[visualize_regression_cam](../vis.visualization/#visualize_cam) do. 
Details on regression visualizations will be covered in a separate section.

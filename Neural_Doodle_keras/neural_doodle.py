'''Neural doodle with Keras

Script usage:
```
python neural_doodle.py nb_colors path_to_style_image.png path_to_style_mask.png \
        [optional_path_to_content_image] \
        path_to_your_doodle.png prefix_for_results
```

Examples:
1. doodle with (1) number of regions (colors) in doodle
(2) a style image, (3) semantic mask of style image,
(4) semantic mask of target image (your doodle)
```
python neural_doodle.py --nlabels 4 --style-image Monet/style.png \
--style-mask Monet/style_mask.png --target-mask Monet/target_mask.png \
--target-image-prefix generated/monet_
```
2. doodle with (1) number of regions (colors) in doodle
(2) a style image, (3) semantic mask of style image,
(4) semantic mask of target_image (your doodle), and
(5) optional content_image that helps generate target
```
python neural_doodle.py --nlabels 4 --style-image Renoir/style.png \
--style-mask Renoir/style_mask.png --target-mask Renoir/target_mask.jpg \
--content-image Renoir/creek.jpg \
--target-image-prefix generated/renoir_
```

References:
[Dmitry Ulyanov's blog on fast-neural-doodle](http://dmitryulyanov.github.io/feed-forward-neural-doodle/)
[Torch code for fast-neural-doodle](https://github.com/DmitryUlyanov/fast-neural-doodle)

Resources:
Images can be downloaded from
https://github.com/DmitryUlyanov/fast-neural-doodle/tree/master/data
'''
from __future__ import print_function
import time
import argparse
import numpy as np
#from scipy.cluster import vq
from sklearn.cluster import k_means
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imread, imsave

np.random.seed(0)

from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19

# Configure and parse parameters
parser = argparse.ArgumentParser(description='neural doodle example with Keras')
parser.add_argument('--nlabels', type=int,
                    help='number of semantic labels (colors) in style_mask/target_mask')
parser.add_argument('--style-image', type=str,
                    help='path to image where style is transfered from')
parser.add_argument('--style-mask', type=str,
                    help='path to semantic mask of style image')
parser.add_argument('--target-mask', type=str,
                    help='path to your doodle image')
parser.add_argument('--content-image', type=str, default=None,
                    help='path to optional content image')
parser.add_argument('--target-image-prefix', type=str,
                    help='prefix for saved result')
args = parser.parse_args()

style_img_path = args.style_image
style_mask_path = args.style_mask
target_mask_path = args.target_mask
content_img_path = args.content_image
target_img_prefix = args.target_image_prefix
use_content_img = content_img_path is not None

nb_labels = args.nlabels
nb_colors = 3 # RGB
# determine image sizes based on target_mask
ref_img = imread(target_mask_path)
img_nrows, img_ncols = ref_img.shape[:2]

total_variation_weight = 5000.
style_weight = 1.
content_weight = 0.025 if use_content_img else 0

content_feature_layers = ['block4_conv2']
style_feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1','block5_conv1']


# helper functions for reading/processing images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_semantic_map():
    target_mask_img = img_to_array(load_img(target_mask_path,
                                target_size=(img_nrows, img_ncols)))
    style_mask_img = img_to_array(load_img(style_mask_path,
                                target_size=(img_nrows, img_ncols)))
    if K.image_dim_ordering() == 'th':
        mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T, 
                              target_mask_img.reshape((3, -1)).T])
    else:
        mask_vecs = np.vstack([style_mask_img.reshape((-1, 3)), 
                              target_mask_img.reshape((-1, 3))])
    #_, labels = vq.kmeans2(mask_vecs, nb_labels, missing='raise')
    _, labels, _ = k_means(mask_vecs.astype("float64"), nb_labels)
    style_mask_label = labels[:img_nrows*img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows*img_ncols:].reshape((img_nrows, img_ncols))
    style_mask = np.stack([style_mask_label==r for r in xrange(nb_labels)], axis = 0)
    target_mask = np.stack([target_mask_label==r for r in xrange(nb_labels)], axis = 0)
    return np.expand_dims(style_mask, axis=0), np.expand_dims(target_mask, axis=0)

# create tensor variables for images
if K.image_dim_ordering() == 'th':
    shape = (1, nb_colors, img_nrows, img_ncols)
else:
    shape = (1, img_nrows, img_ncols, nb_colors)    
style_image = K.variable(preprocess_image(style_img_path))
target_image = K.placeholder(shape = shape)
if use_content_img:
    content_image = K.variable(preprocess_image(content_img_path))
else:
    content_image = K.zeros(shape = shape)
images = K.concatenate([style_image, target_image, content_image], axis=0)

# create tensor variables for masks
raw_style_mask, raw_target_mask = load_semantic_map()
style_mask = K.variable(raw_style_mask.astype("float32"))
target_mask = K.variable(raw_target_mask.astype("float32"))
masks = K.concatenate([style_mask, target_mask], axis=0)

# index constants for images and tasks variables
STYLE, TARGET, CONTENT = 0, 1, 2

# build image model, mask model and target
# vgg19 for image model
image_model = vgg19.VGG19(include_top=False, input_tensor=images)
# pooling layers for mask model
mask_input = Input(tensor=masks, shape=(None, None, None), name="mask_input")
x = mask_input
for layer in image_model.layers[1:]:
    name = 'mask_%s' % layer.name
    if 'conv' in layer.name:
        x = AveragePooling2D((3, 3), strides = (1, 1), name=name, border_mode="same")(x)
        #x = MaxPooling2D((3, 3), strides = (1, 1), name=name, border_mode="same", )(x)
    elif 'pool' in layer.name:
        #x = MaxPooling2D((2, 2), name=name)(x)
        x = AveragePooling2D((2, 2), name=name)(x)
mask_model = Model(mask_input, x)
# collect features from image_model and task_model
image_features = {}
mask_features = {}
for img_layer, mask_layer in zip(image_model.layers, mask_model.layers):
    if 'conv' in img_layer.name:
        assert 'mask_' + img_layer.name == mask_layer.name
        layer_name = img_layer.name
        img_feat, mask_feat = img_layer.output, mask_layer.output
        image_features[layer_name] = img_feat
        mask_features[layer_name] = mask_feat

# define loss functions
# gram matrix helper function
def gram_matrix(x):
    assert 3 == K.ndim(x)
    feats = K.batch_flatten(x)
    gram = K.dot(feats, K.transpose(feats))
    return gram


# style loss in one region
def region_style_loss(style_image, target_image, style_mask, target_mask):
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 2 == K.ndim(style_mask) == K.ndim(target_mask)
    s = gram_matrix(style_image * style_mask) * style_mask.sum()
    c = gram_matrix(target_image * target_mask) * target_mask.sum()
    return K.sum(K.square(s - c)) 


# style loss between style and target images, with
# regions defined in a set of masks
def style_loss(style_image, target_image, style_masks, target_masks):
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 3 == K.ndim(style_masks) == K.ndim(target_masks)
    loss = K.variable(0)
    for i in xrange(nb_labels):
        style_mask = style_masks[i, :, :]
        target_mask = target_masks[i, :, :]
        loss += region_style_loss(style_image, target_image, style_mask, target_mask)
    size = img_nrows * img_ncols
    return loss / (4. * nb_colors**2 * size**2)


# content loss between content and target images
def content_loss(content_image, target_image):
    return K.sum(K.square(target_image - content_image))


# total variance loss for an image
def total_variation_loss(x):
    assert 4 == K.ndim(x)
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, 1:, :img_ncols-1])
        b = K.square(x[:, :, :img_nrows-1, :img_ncols-1] - x[:, :, :img_nrows-1, 1:])
    else:
        a = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
        b = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

loss = K.variable(0)
for layer in content_feature_layers:
    content_feat = image_features[layer][CONTENT, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    loss += content_weight * content_loss(content_feat, target_feat)

for layer in style_feature_layers:
    style_feat = image_features[layer][STYLE, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    style_masks = mask_features[layer][STYLE, :, :, :]
    target_masks = mask_features[layer][TARGET, :, :, :]
    sl = style_loss(style_feat, target_feat, style_masks, target_masks)
    loss += (style_weight / len(style_feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(target_image)
loss_grads = K.gradients(loss, target_image)

# Evaluator class for computing efficiency
outputs = [loss]
if type(loss_grads) in {list, tuple}:
    outputs += loss_grads
else:
    outputs.append(loss_grads)

f_outputs = K.function([target_image], outputs)

def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# Generate images by iterative optimization
if K.image_dim_ordering() == 'th':
    x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128.
else:
    x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

for i in range(60):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = target_img_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
# CNN

CNNs are similar to ordinary NN:
- Neurons with learnable weights and biases
- Each neuron receives inputs, performs dot product, and optionally follows it
  with a non-linearity
- Whole network expresses a single differentiable score function: from raw image
  pixels to class scores
- Loss function (eg, SVM/Softmax) on the last (fully-connected) layer

CNN makes explicity assumption that inputs are images, which allows us to
encode certain properties in the architecture, making them more efficient to
implement and vastly reducing the amount of parameters in the network.

## Architecture Overview

Regular NN
- Receive an input (single vector) transforms it through a series of hidden
  layers
- Each hidden layer is made up of a set of neurons, where each neuron is fully
  connected to all neurons in the previouslayer
- Neurons in a single layer function completely independently and do not share
  any connections
- Last fully-connected layer (output layer) and in classification settings it
  represents the class scores

Regular NN don't scale well to full images

3D volumes of neurons
- CNN's takes advantage of images as input and they constrain the architecture
  in a more sensible way
- Unlike a regular NN, layers of CNN have neurons arranged in 3D (width, height,
  depth).
- Note: depth refers to the 3rd dimension of an activation volume (not the depth
  of a full NN which is the total number of layers in a network)
- Eg, CIFAR-10 images are an input volume of activations, and the volume has
  dimensions 32x32x3 (width x height x depth)
- Neurons in a layer will only be connected to a small region of the layer
  before it, instead of all the neurons in a fully-connected manner
- Final output layer would for CIFAR-10 have 1x1x10, because the CNN architecture
  will reduce the full image into a single vector of class scores, arranged
  along the depth dimension


## CNN Layers

3 Main Types of Layers
- Convolutional Layer
- Pooling Layer
- Fully-Connected Layer (also in regular NN)


Example Architecture (for CIFAR-10)

INPUT [32 x 32 x 3]
- Hold raw pixel values of image

CONVOLUTIONAL LAYER
- Compute the output of neurons that are connected to local regions in the
  input, each computing a dot product between their weights and a small region
  they are connected to in the input volume
- May result in volume such as 32 x 32 x 12 if we decide to use 12 filters

RELU LAYER
- Apply an elementwise activation function, eg, max(0,x) thresholding at zero
- Leaves the size of the volume unchanged (32x32x12)

POOL LAYER
- Perform a downsampling operation along the spatial dimensions (width, height),
  resulting in volume such as [16x16x12]

FULLY CONNECTED LAYER
- Compute the class scores, resulting in volumes of size [1x1x10]
- Where each of the 10 numbers correspond to a class score, such as among the 10
  categories of CIFAR-10
- As with ordinary NN, each neuron in this layer will be connected to all the
  numbers in the previous volume

Summary:
- CNN architecture is in the simplest case a list of Layers that transform the
  image volume into an output volume (ie, holding the class scores)
- CONV, FC, RELU, POOL are by far the most popular
- Each layer accepts an input 3D volume and transforms it into output 3D volume
  thru a differentiable function
- Each layer may or may not have parameters (CONV/FC do, RELU/POOL don't)
- Each layer may or may not have additional hyperparameters (CONV/FC/POOL do,
  RELU doesn't)

### Convolutional Layer

Intuition
- CONV layer parameters consist of a set of learnable filters
- Every filter is small spatially (along width and height), but extends thru the
  full depth of the input volume
- Eg, Typical filter on first layer might have 5x5x3
- During the forward pass, we slide (convolve) each filter across the width and
  height of the input volume and compute the dot products between the entries of
  the filter and the input at any position
- As we slide the filter over the width and height of the input volume we wil
  produce a 2D activation map that gives the responses of that filter at every
  spatial position
- Intuitively, the network will learn filters that activate when they see some
  type of visual feature such as an edge of some orientation or a blotch of some
  color on the first layer, or eventually entire honeycomb or wheel-like
  patterns on higher layers of the network
- Now, we will have an entire set of filters in each CONV layer (eg, 12
  filters), and each will produce a separate 2D activation map
- We will stack these activation maps along the depth dimension and produce the
  output volume

Brain Analogy
- Every entry in the 3D output volume can also be interpreted as an output of
  a neuron that looks at only as small region in the input and shares parameters
  with all neurons to the left and right spatially (since these numbers all
  result from applying the same filter).

Local Connectivity
- With high dimensional inputs like images, cannot have fully connected network,
  so we will connect each neuron to only a local region of ithe input volume
- The spatial extent of this connectivity is a hyperparameter called the
  receptive field of the neuron (equivalently this is the filter size)
- The extent of the connectivity along the depth axis is always equal to the
  depth of the input volume
- Emphasize again the asymmetry in how we treat the spatial dimensions (width
  and height) and along the depth dimension
- Connections are local in space (along width and height), but always full along
  the entire depth of the volume

Spatial Arrangement:  CONV output volume hyperparameters: depth, stride, and zero-padding

DEPTH
- Depth of output volume is hyperparameter; Corresponds to the number of filters
  we would like to use, each learning to look for something different in the
  input
- Eg, if first CONV layer takes as input the raw image, then different neurons
  along the depth dimension may activate in presence of various oriented edged,
  or blobs of color.
- We refer to a set of neurons that are all looking at the same region of the
  input as a depth column

STRIDE
- How the filter slides
- Stride = 1, filter moves one pixel at a time
- Stride = 2, filter jumps two pixels at a time as we slide them around.  This
  will produce smaller output volumes spatially

ZERO-PADDING
- Sometimes convenient to pad the input volume with zeros around the border
- Allows us to control the spatial size of the output volumes
- Most commonly, we will use it to exactly preserve the spatial size of the
  input volume so the input and output width and height are the same



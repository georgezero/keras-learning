# Image Classification


- Assigning an input image one label from a fixed set of categories
- Many other CV tasks (object detection, segmentation) reduced to image
  segmentation

Ex: Image classification model taking a single image to assign probability to
4 labels {cat, dog, hat, mug}
- computer image is 3D array (W x H x RGB)
- So 240 x 400 x 3 image = 297,600 numbers
- Each number is an integer from 0 (black) to 255 (white)
- Task is to take these 297600 numbers into a single label such as 'cat'

Challenges
- Viewpoint variation
- Scale variation
- Deformation (many objects are not rigid)
- Occlusion (only a small portion visible)
- Illumination conditions
- Background clutter (object may blend into their environment)
- Intra-class variation (many types of chairs, etc)

Good classification model must be invariant to the cross product of all these
variations, while simultaneously retaining sensitivity to the inter-class
variations

Data-driven approach
- Give computer many examples of each class
- Training dataset

Image classfication pipeline (Array of pixels (single image) and assign a label)
- Input: set of N images, each labeled with one of K different classes (training
  set)
- Learning: training a classifier / learning a model
- Evaluation: for a new set of images the classifier has never seen before.
  Hope that a lot of predictions match up with true answers (ground truth)

Hyperparameters: distance functions

Training Set
Validation Set (fake test set, small subset of training set)
Test Set (only use once, at the end)

Cross-validation
- Small training data
- Randomly pick validation set and the rest training set
- Try multiple validation sets
- 5-fold cross validation (divide into equal parts; 1 part validation, 4 parts
  training; repeat 5 times changing the validation set)
- Average performance across the 5 different folds

# Linear Classification

Score function - map raw image pixels to class scores (eg, linear function)

Loss function - measures quality of particular set of parameters based on how well the induced scores agreed with ground truth labels in the training data

# Optimization 1

Gradient Descent (Vanilla)

```
while True:
    weights_grad = evaluate_grandient(loss_fun, data, weights)
    weights += - step_size * weights_grad # perform parameter update
```

Simple loop is core of all NN libraries.  Gradient Descent is by far the most common optimization.

Mini-batch Gradient Descent
- Size of mini-batch is a hyperparameter usually based on memory constraints
- Or set some value (eg, 32, 64, 128), use power of 2 because many vectorized operation implementations work faster when their inputs are  sized in powers of 2
```
while True:
    data_batch = sample_training_data(data, 256) # sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += - stepsize * weights_grad # perform parameter updates
```

Stochastic Gradient Descent (SGD)
- Extreme case where mini-batch contains only single example
- AKA on-line gradient descent

# Backpropagation

- Way of computing gradients of expressions through recursive application of chain rule


Problem Statement: Given some function f(x) where x is a vector of inputs and we
are interested in the computing gradient of f at x

Motivation
- f will correspond to the loss function (L) and the inputs x will consist of
  training data and the neural network weights

Gradient calculations

f(x,y) = xy -> df/dx = y and df/dy = x

Compound expressions with chain rule

```
# set some inputs
x = -2; y = 5; z = -4

# perform forward pass
q = x + y # q becomes 3
f = q * z # f becomes -12

# perform backward pass (backprop) in reverse order:
# first backprop through f = q * z
dfdz = q # df/dz = q, so gradient on z becomes 3
dfdq = z # df/dq = z, so gradient on q becomes -4

# now backprop through q = x + y
dfdx = 1.0 * dfdq # dq/dx = 1. And the multiplication here is the chain rule
dfdy = 1.0 * dfdq # dq/dy = 1

This extra multiplcation (for each input) due to the chain rule can turn
a single and relatively useless gate into a cog into a complex circuit such as
an entire neural network

Patterns in backward flow
- add gate
- max gate
- multiply gate

# Neural Networks 1

```
class Neuron(object):
  def forward(inputs):
  """ assume inputs and weights are 1D numpy arrays and bias isa number """
  cell_body_sum = np.sum(inputs * self.weights) + self.bias
  firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid activation
  return firing_rate
```

## Single neuron as a linear classifier

Binary Softmax classifier (AKA logistic regression prediction based on whether
neuron is greater than 5)

Binary SVM

Regularization interpretation

## Commonly used activation functions

Sigmoid

Tanh

*ReLU* - be careful of learning rates and % dead units

Leaky ReLU

Maxout

## NN Architectures

Most common layer type is fully-connected layer

Naming conventions: N-layer neural network (exclude input layer)

Output layer: Most commonly doesn't have activation function

Sizing:
- Number of neurons
- Number of parameters (weights + biases)

## Example

```
# forward-pass of a 3-layer neural network
f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
x = np.random.randn(3, 1) # random input vector of three numbers (3x1)
h1 = f(np.dot(W1, x) + b1) # calculate 1st hidden layer activations (4x1)
h2 = f(np.dot(W2, h1) + b2) # calculate 2nd hidden layer activations (4x1)
out = np.dot(W3, h2) + b3 # output neuron (1x1)
```

# NN 2

## Setting up the data and the model

Data Preprocessing

- Mean subtraction
- Normalization
- PCA and whitening

Mean must only be computed on the training data, and then applied to the
validation / test data

## Weight Initialization

Pitfall: all zero initialization

Small random numbers

Calibrating the variances with 1/sqrt(n)

Sparse initialization

Initializing the biases

IN PRACTICE: current recommendation is to use ReLU units and use the

wp = np.random.randn(n) * sqrt(2.0/n)


Batch Normalization

## Regularization

L2 regularization
L1 regularization
Max norm constraints
Dropout


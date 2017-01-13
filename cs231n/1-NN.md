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

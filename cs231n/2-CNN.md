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



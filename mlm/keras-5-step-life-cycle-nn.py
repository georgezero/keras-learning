"""
http://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/

# 5 steps in NN model lifecycle in keras

1. Define network
2. Compile network
3. Fit network
4. Evaluate network
5. Make predictions

# Step 1: Define network

NN are defined in keras as a sequence of layers -- contained in the Sequential
class

First step is to create an instance of the Sequential class.  Then create layers and add them in the order that they should be connected
"""

model = Sequential()
model.add(Dense(2))

"""
Can also do this in one step by creating an array of layers and passing it to the constructor of the Sequential 
"""

layers = [Dense(2)]
model = Sequential(layers)

"""
First layer in the network must define the number of inputs to expect.

For a multilayer perceptron model, this is specified by the input_dim attribute.

Small MLP with 2 inputs in visible layer, 5 neurons in hidden layer, and 1 neuron in the output layer can be defined as:
"""
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(1))

"""
Think of Sequential model as a pipeline with your raw data fed in at the bottom and predictions that come out at the top

Activation functions that transform a summed signal from each neuron layer can be extracted and added to the Sequential layer-like object called Activation.
"""

model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

"""
The choice of activation function is most important for the output layer as it will define the format that predictions will take.

For example, below are some common predictive modeling problem types and the structure and standard activation function that you can use in the output layer:

    Regression: Linear activation function or 'linear' and the number of neurons matching the # outputs
    Binary Classification (2 class): Logistic activation or 'sigmoid' and one neuron in the output layer
    Multiclass Classificaton (> 2 class): Softmax activation function or 'softmax' and one output neuron per class value, assuming a one-hot encoded output pattern.


"""

# Step 2: Compile Network


"""
Once we have defined our network, we must compile it.

Compilation is an efficient step. It transforms the simple sequence of layers that we defined into a highly efficient series of matrix transformations in a format intended to be executed on your GPU / CPU depending on how keras is configured.

Think of compilation as a precompute step for your network.

Compilation is always required after defining a model.  This includes both before training it using an optimization scheme as well as loading a set of pre-trained weights from a save file.
The reason is that the compilation step prepares an efficient representation of the network that is also required to make predictions on your hardware.


"""


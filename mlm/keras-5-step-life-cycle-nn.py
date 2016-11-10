"""
http://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/

# 5 steps in NN model lifecycle in keras

1. Define network
2. Compile network
3. Fit network
4. Evaluate network
5. Make predictions

# Define network

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





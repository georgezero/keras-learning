
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# or use .add()

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

###
# Specify input shape
### only for first layer, because following layers can do automatic shape inference
### input_shape, batch_input_shape, input_dim (Dense)

model = Sequential()
model.add(Dense(32, input_shape(784,)))

# equivalent to:

model = Sequential()
model.add(Dense(32, batch_input_shape=(None, 784)))
# note that batch dimension is 'None'
# so the model will be able to process batches of any size

# equivalent to:

model = Sequential()
model.add(Dense(32, input_dim=784))

# And these are equivalent:
# 1
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# 2
model = Sequential()
model.add(LSTM(32, batch_input_shape=(None, 10, 64)))
# 3
model = Sequential()
model.add(LSTM(32, input_length=10, input_dim=64))


###
# Merge layer

# Multiple Sequential instances can be merged into a single output via a Merge layer.  The output is a layer that can be added as first layer in a new Sequential model:

from keras.layers import Merge

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

final_model = Sequential()
final_model.add(merged) # first layer in new Sequential model
final_model.add(Dense(10, activation='softmax'))

# This two-branch model can then be trained via, eg:

final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
final_model.fit([input_data_1, input_data_2], targets) # pass one data array per model input

# Now you know enough to be able to define almost any model with Keras. For complex models that cannot be expressed via Sequential and Merge, you can use the functional API.

###
# Compilation

# Before training a model, you need to configure the learning process, which is done via the compile method. It receives three arguments:

# an optimizer. This could be the string identifier of an existing optimizer (such as  rmsprop or adagrad), or an instance of the Optimizer class. See: optimizers.
#a loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or  mse), or it can be an objective function. See: objectives.
# a list of metrics. For any classification problem you will want to set this to  metrics=['accuracy']. A metric could be the string identifier of an existing metric or a custom metric function. Custom metric function should return either a single tensor value or a dict metric_name -> metric_value. See: metrics.

# for a multi-class classification problem
model.compile(optimizer='rmsprop',
        loss='categorical_corssentropy',
        metrics=['accuracy'])

# for a binary classification problem
model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

# for a mean squared error regression problem
model.compile(optimizer='rmsprop',
        loss='mse')

# for custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def false_rates(y_true, y_pred):
    false_neg = ...
    false_pos = ...
    return {
        'false_neg': false_neg,
        'false_pos': false_pos,
    }

model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy', mean_pred, false_rates])

###
# Training

# Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the fit function. Read its documentation here.

# for a single-input model with 2 classes (binary):

model = Sequential()
model.add(Dense(1, input_dim=784, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 1))

# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)

# for a multi-input model with 10 classes:

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# generate dummy data
import numpy as np
from keras.utils.np_utils import to_categorical
data_1 = np.random.random((1000, 784))
data_2 = np.random.random((1000, 784))

# these are integers between 0 and 9
labels = np.random.randint(10, size=(1000, 1))
# we convert the labels to a binary matrix of size (1000, 10)
# for use with categorical_crossentropy
labels = to_categorical(labels, 10)

# train the model
# note that we are passing a list of Numpy arrays as training data
# since the model has 2 inputs
model.fit([data_1, data_2], labels, nb_epoch=10, batch_size=32)



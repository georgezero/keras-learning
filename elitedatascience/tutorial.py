import numpy as np
np.random.seed(123) # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# Load MNIST data
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print X_train.shape
# (60000, 28, 28)

# from matplotlib import pyplot as plt
# plt.imshow(X_train[0])

# convert data type to float32 and normalize data to range [0,1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels for Keras

# print y_train.shape
# (60000,)

# print y_train[:10]
# [5 0 4 1 9 2 1 3 1 4]


# Convert 1D class arrays to 10D class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# print Y_train.shape
# (60000, 10)

# Define model architecture

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))

# print model.output_shape
# (None, 32, 26, 26)

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))




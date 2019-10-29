from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import LeakyReLU
from keras.utils import np_utils
import numpy as np

def LeNet_mnist(num_output = 10):
    # LeNet model:
    # not using Relu,but Leaky Relu

    # Create a sequential model
    model = Sequential()

    # Add the first convolution layer
    model.add(Convolution2D(
        filters = 20,
        kernel_size = (5, 5),
        padding = "same",
        input_shape = (28, 28, 1)))

    # Add a ReLU activation function
#     model.add(Activation(
#         activation = "relu"))
    model.add(LeakyReLU())

    # Add a pooling layer
    model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides =  (2, 2)))

    # Add the second convolution layer
    model.add(Convolution2D(
        filters = 50,
        kernel_size = (5, 5),
        padding = "same"))

    # Add a ReLU activation function
#     model.add(Activation(
#         activation = "relu"))
    model.add(LeakyReLU())

    # Add a second pooling layer
    model.add(MaxPooling2D(
        pool_size = (2, 2),
        strides = (2, 2)))

    # Flatten the network
    model.add(Flatten())

    # Add a fully-connected hidden layer
    model.add(Dense(500))

    # Add a ReLU activation function
#     model.add(Activation(
#         activation = "relu"))
    model.add(LeakyReLU())

    # Add a fully-connected output layer
    model.add(Dense(num_output))

    
#     model.add(Lambda(lambda x: x / 0.0001))

#     Add a softmax activation function
    model.add(Activation("softmax"))

# #     reshaped_lenetout = K.reshape(lenetModel.output,(-1,20, 4)) 
#     model.add(Reshape((-1, 20, 4)))


#     def grad_loss(y_true, y_pred):
#         return y_true - y_pred

    
    # Compile the network
    model.compile(
#         loss = "mean_squared_error",
        loss = "categorical_crossentropy",
        
#         loss = grad_loss,
        optimizer = SGD(lr = 0.01),
        metrics = ["accuracy"])
    
    return model
from Layer import Layer
import numpy as np

# This is a sub class of Layer
# It is merely acts as a function which each dense layer is run through
class Activation(Layer):
    #* activation_prime is the derivative of activation
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime


    def forward(self, input):
        self.input = input
        return self.activation(self.input)


    #* Another hard method using derivative in YT video
    def backward(self, output_gradient, learning_rate):
        # StackOverflow - difference between np.dot and np.multiply
        return np.multiply(output_gradient, self.activation_prime(self.input))
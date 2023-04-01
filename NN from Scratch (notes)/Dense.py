from Layer import Layer
import numpy as np

# This is a sub class of Layer
# It creates the dense (hidden) layers between the input layer and the output layer
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)


    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    #* The most difficult part, uses 3 derivative formulas which YT videos explain (I still learned it, but will likely forget how they were found)
    def backward(self, output_gradient, learning_rate):
        #* define the gradients (collection of partial derivatives) for the weight and bias parameters
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = output_gradient
        
        #* update weight and bias parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

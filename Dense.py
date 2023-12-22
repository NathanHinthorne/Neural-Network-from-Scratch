from layer import Layer
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

    # Backward pass
    # The most difficult part. Uses 3 derivative formulas
    def backward(self, output_gradient, learning_rate):

        # The output_gradient is the derivative of the loss function with respect to the output of this layer.
        # This is passed in from the next layer during backpropagation.

        # The weights_gradient is the derivative of the loss function with respect to the weights.
        # It's calculated as the dot product of the output_gradient and the transpose of the input.
        # This is derived from the chain rule of calculus, which states that the derivative of a composite function is
        # the product of the derivative of the outer function and the derivative of the inner function.
        # Here, the outer function is the loss function and the inner function is the output of this layer.
        # Define the gradient (collection of partial derivatives) for the weight and bias parameters
        # Weights gradient: dW = dL/dY * dY/dW = dL/dY * X^T
        weights_gradient = np.dot(output_gradient, self.input.T)

        # The bias_gradient is the derivative of the loss function with respect to the biases.
        # It's simply the output_gradient because the derivative of the output with respect to the biases is 1.
        # Bias gradient: dB = dL/dY * dY/dB = dL/dY
        bias_gradient = output_gradient
        

        # The weights and biases are updated using gradient descent.
        # The learning_rate is a hyperparameter that determines the step size when updating the weights and biases.
        # The weights and biases are updated in the direction that minimizes the loss function, which is why the gradients
        # are subtracted from the weights and biases.

        # New weights: W_new = W_old - learning_rate * dW
        self.weights -= learning_rate * weights_gradient
        # New biases: B_new = B_old - learning_rate * dB
        self.bias -= learning_rate * bias_gradient

        # The input_gradient is the derivative of the loss function with respect to the input of this layer.
        # It's calculated as the dot product of the transpose of the weights and the output_gradient.
        # This is derived from the chain rule of calculus, which states that the derivative of a composite function is
        # the product of the derivative of the outer function and the derivative of the inner function.
        # Here, the outer function is the loss function and the inner function is the output of this layer.
        # Input gradient: dX = dL/dY * dY/dX = W^T * dL/dY
        return np.dot(self.weights.T, output_gradient)

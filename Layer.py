# This is the super class for Activation and Dense
# It defines a general layer that can be extended from to create the input, dense, and output layers

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


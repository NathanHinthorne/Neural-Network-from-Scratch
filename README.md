## NOTE 
This code is adapted from https://www.youtube.com/watch?v=pauPCy_s0Ok - "The Independent Code"

Although I followed the video closely, I made sure to pause and learn the details and reasoning behind every single formula, which often required hours of further research on the subject.


## What I Learned

### Gradients and Partial Derivatives

In the context of machine learning and neural networks, a gradient is a vector that contains the partial derivatives of a function with respect to its variables. It's a direction in which the function increases most rapidly from a particular point.

A partial derivative of a function of several variables is its derivative with respect to one of those variables, with the others held constant. For example, if you have a function `f(x, y)`, the partial derivative of `f` with respect to `x` is denoted as `∂f/∂x` or `f_x`, and it measures the rate at which the function changes with respect to `x` when `y` is held constant.

### How Neural Networks Learn

Neural networks learn by updating their weights and biases in the direction that minimizes the loss function. This process is known as gradient descent. The gradient points in the direction of the steepest ascent in the error surface, and the weights and biases are updated in the opposite direction of the gradient to minimize the error.

### Q & A

**Q: How is the gradient used in neural networks?**

A: The gradient is used to update the weights and biases of the neural network during the backpropagation step of training. The gradient points in the direction of the steepest ascent in the error surface, and the weights and biases are updated in the opposite direction of the gradient to minimize the error.

**Q: What is the role of the learning rate in neural networks?**

A: The learning rate is a hyperparameter that determines the step size when updating the weights and biases. A smaller learning rate means that the network will learn slowly, while a larger learning rate means that the network may learn quickly, but it may also overshoot the optimal solution.

**Q: How are the weights and biases updated in a neural network?**

A: The weights and biases are updated using the gradients and the learning rate. The weights and biases are updated in the direction that minimizes the loss function, which is why the gradients are subtracted from the weights and biases.

**Q: What is the chain rule and why is it important in neural networks?**

A: The chain rule is a method for finding the derivative of composite functions, or functions that are made by combining one or more functions. It's crucial in neural networks because it allows us to compute the derivative of the loss function with respect to the weights and biases, which is necessary for updating the weights and biases during training.

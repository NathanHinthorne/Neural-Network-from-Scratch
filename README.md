## NOTE 
This code is adapted from [The Independent Code](https://www.youtube.com/watch?v=pauPCy_s0Ok). While I used the video for reference, I took the time to pause and delve into the details and reasoning behind every formula. This often required hours of further research on the subject. I pondered what it all meant and got an intuitive feel for the math which occurs behind the scenes in neural networks.

<br />


# What I Learned
Below is a summary of my core takeaways from the completion of this project and extensive research I did for it.

### Gradients and Partial Derivatives
In machine learning and neural networks, a gradient is a vector that holds the partial derivatives of a function with respect to its variables. It points in the direction where the function increases most rapidly.

A partial derivative of a function with several variables is its derivative with respect to one of those variables, while the others are held constant. For instance, if you have a function `f(x, y)`, the partial derivative of `f` with respect to `x` (denoted as `∂f/∂x` or `f_x`) measures the rate at which the function changes with respect to `x`, assuming `y` is constant.

To visualize the gradient, imagine a 3D shape sliced by a plane on the x-axis and then on the y-axis. The intersection between the shape's surface and each plane forms two distinct curves. The slopes of these curves' tangent lines are plugged into the gradient vector as its components.

### How Neural Networks Learn
Neural networks learn by adjusting their weights and biases to minimize the loss function, a process known as gradient descent. The gradient points in the direction of steepest ascent on the error surface, and the weights and biases are updated in the opposite direction of the gradient to reduce the error.

We don't have a complete view of the cost function. We use the results from each forward propagation of the cost function to determine the direction to step in. While the cost function outputs a single value indicating the neural network's performance, the backpropagation process uses the cost function's gradient with respect to the weights. This gradient provides a wealth of information about how to adjust each weight in the network to enhance performance. Each partial derivative in the gradient indicates how much the cost function changes when the corresponding weight is tweaked slightly. By calculating the gradient, we get a direction in the high-dimensional space of weights. Adjusting the weights in the opposite direction of the gradient allows us to decrease the cost function as quickly as possible. This is why a single value (the cost function) is sufficient to improve the entire network of neurons.

### Formulas Used in Back-propagation
1. The weights gradient is the derivative of the loss function with respect to the weights. It's calculated as the dot product of the output gradient and the transpose of the input. This is derived from the chain rule of calculus, which states that the derivative of a composite function is the product of the derivative of the outer function and the derivative of the inner function. Here, the outer function is the loss function and the inner function is the output of this layer.

    $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot X^t$$

2. The bias gradient is the derivative of the loss function with respect to the biases. It's simply the output gradient because the derivative of the output with respect to the biases is 1. This means that the rate of change of the loss function with respect to the biases is the same as the rate of change of the loss function with respect to the output.

    $$\frac{\partial L}{\partial B} =\frac{\partial L}{\partial Y}$$

3. The input gradient is the derivative of the loss function with respect to the input. It's calculated as the dot product of the transpose of the weights and the output gradient. This is also derived from the chain rule of calculus. The input gradient tells us how much the loss function would change if we changed the inputs slightly.

    $$\frac{\partial L}{\partial X} = W^t \cdot \frac{\partial L}{\partial Y}$$

# Q & A

**Q: How is the gradient used in neural networks?**

A: In neural networks, the gradient is used during the backpropagation step of training. It points in the direction of steepest ascent on the error surface. The weights and biases are updated in the opposite direction of the gradient, which minimizes the error and improves the network's performance.

<br />

**Q: Why does the gradient always point in the direction of steepest ascent?**

A: The gradient points in the direction of steepest ascent because of how it's calculated. To visualize this, consider a 3D shape. Imagine slicing this shape first with a plane on the x-axis, then with a plane on the y-axis. The intersection between the shape's surface and each plane forms two distinct curves. The tangent lines of these curves represent the steepest slope at that point on each curve. 

When these slopes are used as the components of the gradient vector, they inherently capture the direction of steepest ascent at that point. This is because the tangent lines represent the maximum rate of change of the function at that point along each axis. Therefore, a vector composed of these maximum rates of change along each axis will naturally point in the direction where the function increases most rapidly, which is the direction of steepest ascent.

<br />

**Q: How do we know how to calculate the gradient of the cost function since we don't even know what the cost function looks like?**

A: We don't need a complete view of the cost function to calculate its gradient. We use the results from each forward propagation of the cost function to determine the direction to step in. The gradient of the cost function with respect to the weights gives us a direction in the high-dimensional space of weights. Adjusting the weights in the opposite direction of the gradient allows us to decrease the cost function as quickly as possible.

<br />

**Q: How does the cost function give enough info to modify weights? Doesn't the cost function just output a single value? Why is one value enough to improve the WHOLE network of neurons?**


A: While the cost function outputs a single value indicating the neural network's performance, the backpropagation process uses the cost function's gradient with respect to the weights. This gradient provides a wealth of information about how to adjust each weight in the network to enhance performance. Each partial derivative in the gradient indicates how much the cost function changes when the corresponding weight is tweaked slightly. By calculating the gradient, we get a direction in the high-dimensional space of weights. Adjusting the weights in the opposite direction of the gradient allows us to decrease the cost function as quickly as possible. This is why a single value (the cost function) is sufficient to improve the entire network of neurons.

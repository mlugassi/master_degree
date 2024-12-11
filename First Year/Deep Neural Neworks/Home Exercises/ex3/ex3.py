
import numpy as np
import matplotlib.pyplot as plt

"""
Exercise 0: Basic Gradient Descent Implementation
------------------------------------------------
In this exercise, you'll implement basic gradient descent for two functions:
x² and x⁴. This will help you understand the fundamental concepts of optimization.
"""


def x2(x):
    """Function that returns x²"""
    return x * x


def x2_(x):
    """Derivative of x²
    TODO: Implement the derivative of x²
    Hint: The derivative of x² is 2x
    """
    return 2 * x


def x4(x):
    """Function that returns x⁴"""
    return x ** 4


def x4_(x):
    """Derivative of x⁴
    TODO: Implement the derivative of x⁴
    Hint: The derivative of x⁴ is 4x³
    """
    return 4 * (x ** 3)


"""
Exercise 1: Momentum Method Implementation
----------------------------------------
Compare the convergence of standard gradient descent with the momentum method
for both x² and x⁴ functions.
"""


def momentum_update(velocity, gradient, momentum=0.9):
    """
    TODO: Implement the momentum update
    Args:
        velocity: Current velocity
        gradient: Current gradient
        momentum: Momentum coefficient (default: 0.9)
    Returns:
        Updated velocity
    """
    return None  # Replace with your implementation


"""
Exercise 2: Advanced Optimization Methods
---------------------------------------
Implement and compare four different optimization methods:
1. Gradient Descent (SGD)
2. Momentum
3. Nesterov Accelerated Gradient (NAG)
4. AdaGrad
"""


def nesterov_update(x, velocity, gradient_func, momentum=0.9):
    """
    TODO: Implement Nesterov update
    Hint: Look ahead using current velocity before computing gradient
    """
    return None  # Replace with your implementation


def adagrad_update(gradient, historical_gradient):
    """
    TODO: Implement AdaGrad update
    Hint: Use historical gradient to adjust learning rate
    """
    return None  # Replace with your implementation


# Example usage and testing code
if __name__ == "__main__":
    # Starting points
    X2 = X4 = X2m = X4m = X2n = X4n = X2g = X4g = 10.0

    # Hyperparameters (you should tune these)
    lr = 0.01  # Learning rate for basic gradient descent
    momentum = 0.9  # Momentum coefficient
    num_steps = 100  # Number of optimization steps

    # Storage for plotting
    history = {
        'sgd_x2': [], 'sgd_x4': [],
        'momentum_x2': [], 'momentum_x4': [],
        'nag_x2': [], 'nag_x4': [],
        'adagrad_x2': [], 'adagrad_x4': []
    }

    # TODO: Implement the training loops for each method

    # TODO: Create visualization of convergence paths

    # TODO: Compare and analyze the results

"""
Submission Requirements:
1. Implement all TODO sections
2. Include visualizations comparing the convergence of different methods
3. Write a brief analysis (1-2 paragraphs) explaining your observations
4. Submit your code and a brief report in PDF format



"""
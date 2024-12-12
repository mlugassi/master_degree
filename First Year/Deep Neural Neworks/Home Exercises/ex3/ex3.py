
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
    return np.power(x, 2)


def x2_(x):
    """Derivative of x²
    TODO: Implement the derivative of x²
    Hint: The derivative of x² is 2x
    """
    return 2 * x


def x4(x):
    """Function that returns x⁴"""
    return np.power(x, 4)


def x4_(x):
    """Derivative of x⁴
    TODO: Implement the derivative of x⁴
    Hint: The derivative of x⁴ is 4x³
    """
    return 4 * np.power(x, 3)


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
    new_velocity = (velocity * momentum) - gradient
    return new_velocity


"""
Exercise 2: Advanced Optimization Methods
---------------------------------------
Implement and compare four different optimization methods:
1. Gradient Descent (SGD)
2. Momentum
3. Nesterov Accelerated Gradient (NAG)
4. AdaGrad
"""
9
def calc_gradient(dfx, lr):
    return lr * dfx

def gradient_descent_update(x, dfx, lr):
    return x - calc_gradient(dfx, lr)

def nesterov_update(x, velocity, gradient_func, derivative_func, momentum=0.9):
    """
    TODO: Implement Nesterov update
    Hint: Look ahead using current velocity before computing gradient
    """
    x_ahead = x + (velocity * momentum)
    new_velocity = (velocity * momentum) - gradient_func(derivative_func(x_ahead), lr)

    return new_velocity


def adagrad_update(gradient, historical_gradient):
    """
    TODO: Implement AdaGrad update
    Hint: Use historical gradient to adjust learning rate
    """
    return - (gradient / np.sqrt(historical_gradient) + 1e-7)


# Example usage and testing code
if __name__ == "__main__":
    # Starting points
    X2 = X4 = X2m = X4m = X2n = X4n = X2g = X4g = 10.0
    X2m_velocity = X4m_velocity = X2n_velocity = X4n_velocity = 0
    X2g_grad_squred = X4g_grad_squred = 0

    # Hyperparameters (you should tune these)
    lr = 0.001  # Learning rate for basic gradient descent
    momentum = 0.5 # Momentum coefficient
    num_steps = 10000  # Number of optimization steps

    # Storage for plotting
    history = {
        'sgd_x2': [X2], 'sgd_x4': [X4],
        'momentum_x2': [X2m], 'momentum_x4': [X4m],
        'nag_x2': [X2n], 'nag_x4': [X4n],
        'adagrad_x2': [X2g], 'adagrad_x4': [X2g]
    }

    for i in range(num_steps):
        X2 = gradient_descent_update(X2, x2_(X2), lr)
        X4 = gradient_descent_update(X4, x4_(X4), lr)

        X2m_velocity = momentum_update(X2m_velocity, calc_gradient(x2_(X2m), lr), momentum)
        X4m_velocity = momentum_update(X4m_velocity, calc_gradient(x4_(X4m), lr), momentum)
        X2m += X2m_velocity
        X4m += X4m_velocity

        X2n_velocity = nesterov_update(X2n, X2n_velocity, calc_gradient, x2_, momentum)
        X4n_velocity = nesterov_update(X4n, X4n_velocity, calc_gradient, x4_, momentum)
        X2n += X2n_velocity
        X4n += X4n_velocity

        X2g_grad_squred += np.power(x2_(X2g), 2)
        X4g_grad_squred += np.power(x4_(X4g), 2)
        X2g += adagrad_update(calc_gradient(x2_(X2g), lr), X2g_grad_squred)
        X4g += adagrad_update(calc_gradient(x4_(X4g), lr), X4g_grad_squred)

        history['sgd_x2'].append(X2)
        history['sgd_x4'].append(X4)
        history['momentum_x2'].append(X2m)
        history['momentum_x4'].append(X4m)
        history['nag_x2'].append(X2n)
        history['nag_x4'].append(X4n)
        history['adagrad_x2'].append(X2g)
        history['adagrad_x4'].append(X4g)

    # TODO: Implement the training loops for each method

    # TODO: Create visualization of convergence paths

    # TODO: Compare and analyze the results

    # Generate the plots
    methods = ['sgd', 'momentum', 'nag', 'adagrad']
    colors = {'sgd': 'blue', 'momentum': 'green', 'nag': 'red', 'adagrad': 'purple'}

    # Create one figure with two subplots
    plt.figure(figsize=(12, 6))

    # Subplot for x^2
    plt.subplot(1, 2, 1)
    for method in methods:
        plt.plot(history[f"{method}_x2"], label=method.upper(), color=colors[method])
    plt.title(f"Covergence for $x^2$ with learning rate = {lr}, mu = {momentum}")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Subplot for x^4
    plt.subplot(1, 2, 2)
    for method in methods:
        plt.plot(history[f"{method}_x4"], label=method.upper(), color=colors[method])
    plt.title(f"Covergence for $x^4$ with learning rate = {lr}, mu = {momentum}")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Show the single figure with both subplots
    plt.tight_layout()
    plt.show()
"""
Submission Requirements:
1. Implement all TODO sections
2. Include visualizations comparing the convergence of different methods
3. Write a brief analysis (1-2 paragraphs) explaining your observations
4. Submit your code and a brief report in PDF format



"""
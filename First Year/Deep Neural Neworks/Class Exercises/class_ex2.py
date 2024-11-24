# Autor Michael Lugassi


#  Step 1: Basic Setup
"""
First, we'll set up our basic imports and create a simple class structure.
Student Task: Import numpy and create an empty Perceptron class
"""
import numpy as np


class Perceptron:
    def __init__(self, size=3, learning_rate=0.1, threshold=0.5):
        self.weights = np.zeros(size)
        self.learning_rate = learning_rate
        self.threshold = threshold

    # Step 2: Implement the Prediction Function
    """
    Now implement the function that calculates the sum and makes a prediction.
    Formula: sum = x0*w0 + x1*w1 + x2*w2
    If sum > threshold, predict 1; otherwise predict 0
    """

    def predict(self, inputs):
        weighted_input = np.dot(inputs, self.weights)

        if weighted_input > self.threshold:
            return 1
        return 0

    def train_step(self, inputs, desired_output):
        prediction = self.predict(inputs)
        err = desired_output - prediction
        correction = err * self.learning_rate  # aaa
        self.weights = self.weights + (correction * inputs)
        return {"prediction": prediction, "error": err, "weights": self.weights}


# Step 4: Complete Implementation
"""
Here's the complete implementation for reference
"""


# Step 5: Training Data Setup
"""
Create training data for the logic function you want to learn
Example: Learning OR function
"""


def create_training_data():
    # Student Task:
    # Create training data as list of tuples: (inputs, desired_output)
    # Remember to include bias input (x0) as 1
    # Format: [(x0, x1, x2), desired_output]
    training_data = [
        ([1, 0, 0], 1),  # Example: First row from your table
        ([1, 0, 1], 1),  # Add more rows...
    ]
    return training_data


# Step 7: Example Usage and Visualization
def main():
    # Create training data
    training_data = [([1, 0, 0], 1), ([1, 0, 1], 1), ([1, 1, 0], 1), ([1, 1, 1], 0)]

    # Create and train perceptron
    p = Perceptron()

    # Training loop
    print("Training Process:")
    print("=" * 50)

    for epoch in range(4):
        print(f"\nEpoch {epoch + 1}")
        for inputs, desired in training_data:
            result = p.train_step(np.array(inputs), desired)

            # Print training step details
            print(f"\nInputs: {inputs}")
            print(f"Desired Output: {desired}")
            print(f"Prediction: {result['prediction']}")
            print(f"Error: {result['error']}")
            print(f"Updated Weights: {result['weights']}")


# Run the example
if __name__ == "__main__":
    main()

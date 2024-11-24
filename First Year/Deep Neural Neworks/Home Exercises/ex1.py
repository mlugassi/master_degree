# Autor Michael Lugassi

import csv
import os
import random
import numpy as np
import matplotlib.pyplot as plt

#  Step 1: Basic Setup
"""
First, we'll set up our basic imports and create a simple class structure.
Student Task: Import numpy and create an empty Perceptron class
"""


class Perceptron:
    """
    A simple implementation of a perceptron model.
    
    Attributes:
        weight (np.ndarray): The weights of the perceptron.
        learning_rate (float): The learning rate for updating weights.
        threshold (float): The threshold to determine the output.
        predicted_outputs (list): A list of predicted outputs during training.
        desired_outputs (list): A list of desired outputs during training.
    """    
    def __init__(self, size=3, learning_rate=0.1, threshold=0.5):
        """
        Initializes the perceptron with weights, learning rate, and threshold.
        
        Args:
            size (int): Number of inputs (including bias).
            learning_rate (float): Learning rate for weight updates.
            threshold (float): The threshold value for predictions.
        """        
        self.weight = np.zeros(size)
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.predicted_outputs = list()
        self.desired_outputs = list()

    def predict(self, inputs):
        """
        Predicts the output for given inputs based on the perceptron weights.
        
        Args:
            inputs (np.ndarray): Input vector including bias.
        
        Returns:
            int: Predicted output (0 or 1).
        """
        res = np.dot(inputs, self.weight)

        if res > self.threshold:
            return 1
        return 0

    def train_step(self, inputs, desired_output):
        """
        Performs one step of training, updating weights based on the error.
        
        Args:
            inputs (np.ndarray): Input vector including bias.
            desired_output (int): The desired output for the given inputs.
        
        Returns:
            dict: A dictionary containing prediction, error, weights, and loss.
        """
        prediction = self.predict(inputs)
        err = desired_output - prediction
        correction = err * self.learning_rate
        self.weight = self.weight + (correction * inputs)
        self.desired_outputs.append(desired_output)
        self.predicted_outputs.append(prediction)
        return {"prediction": prediction, "error": err, "weights": self.weight}


def split_data(dataset, train_percentage=0.8):
    """
    Splits the dataset into training and testing sets.
    
    Args:
        dataset (list): The complete dataset (list of tuples with inputs and desired outputs).
        train_percentage (float): The proportion of data to use for training (0-1 range).
    
    Returns:
        tuple: Training data and testing data.
    """    
    if not 0 <= train_percentage <= 1:
        raise ValueError("train_percentage must be between 0 and 1.")

    # Calculate the number of items for training and testing
    total = len(dataset)
    train_count = round(total * train_percentage)

    # Split the data
    training_data = dataset[:train_count]
    testing_data = dataset[train_count:]
    print("=" * 50)
    print("Loading Data Process:")
    print("=" * 50)
    print(f"Total Traning Data: {len(training_data)} ({round((len(training_data)/len(dataset)) * 100 ,2)}% of dataset)")
    print(f"Total Testing Data: {len(testing_data)} ({round((len(testing_data)/len(dataset)) * 100 ,2)}% of dataset)")
    print("=" * 50)
    return training_data, testing_data


def get_dataset(file_path, shuffle):
    """
    Reads the dataset from a CSV file and processes it into inputs and outputs.
    
    Args:
        file_path (str): Path to the dataset file.
        shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
        tuple: Processed dataset and input size.
    """
    dataset = list()
    # Open the file and read the content
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        # Loop through each row
        for row in reader:
            inputs = [int(num) for num in row[:-1]] #Covert string/char to binary input
            desired_output = 0 if row[-1] == "g1" else 1 #Covert string/char to binary input
            inputs.insert(0, 1) #Add bias with fixed value of 1 to the first index of each inputs
            dataset.append((inputs, desired_output))
    if shuffle:
        random.shuffle(dataset)
    return dataset, len(dataset[0][0])

def show_loss_graph(losses):
    epochs = list(range(1, len(losses) + 1))
    # Create and display the plot
    plt.plot(epochs, losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(1, len(epochs))
    plt.ylim(0, 1)
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def mean_squared_error(desired_outputs, predicted_outputs):
    """
    Calculates the Mean Squared Error (MSE) between the desired and predicted outputs.
    
    Returns:
        float: The mean squared error.
    """ 
    return np.mean((np.array(desired_outputs) - np.array(predicted_outputs)) ** 2)
    
def train_perceptron(perceptron: Perceptron, training_data, epochs=4, print_step_details=True):
    """
    Trains the perceptron using the provided training data over multiple epochs.
    
    Args:
        perceptron (Perceptron): The perceptron model to train.
        training_data (list): The training data (list of tuples with inputs and desired outputs).
        epochs (int): The number of epochs to train for.
        print_step_details (bool): Whether to print detailed information for each step.
    """
    # Training loop
    print("Training Process:")
    print("=" * 50)
    losses_per_epoch = list()
    desired_outputs = list()
    predicted_outputs = list()

    for epoch in range(epochs):
        desired_outputs.clear()
        predicted_outputs.clear()

        if print_step_details:
            print(f"\nEpoch {epoch + 1}")
        for inputs, desired in training_data:
            result = perceptron.train_step(np.array(inputs), desired)
            desired_outputs.append(desired)
            predicted_outputs.append(result['prediction'])
            if print_step_details:
                # Print training step details
                print(f"\nInputs: {inputs}")
                print(f"Desired Output: {desired}")
                print(f"Prediction: {result['prediction']}")
                print(f"Error: {result['error']}")
                print(f"Updated Weights: {result['weights']}")
        losses_per_epoch.append(mean_squared_error(desired_outputs, predicted_outputs))

    show_loss_graph(losses_per_epoch)
    print(f'Total Epochs: {epochs}')
    print(f'Model Loss: {losses_per_epoch[-1]}')
    print(f'Model Weights: {result["weights"]}')
    print("=" * 50)


def check_perceptron(perceptron: Perceptron, checking_data):
    """
    Evaluates the trained perceptron on the checking data and calculates recall and precision.
    
    Args:
        perceptron (Perceptron): The trained perceptron model.
        checking_data (list): The checking data (list of tuples with inputs and desired outputs).
    """
    print("Checking Process:")
    print("=" * 50)
    correct_predictions = 0
    true_pos = 0
    false_neg = 0

    for inputs, desired in checking_data:
        #Checking perceptron prediction and gathering data to evaluate the model's quality
        prediction = perceptron.predict(np.array(inputs))
        error = desired - prediction
        if error == 0:
            correct_predictions += 1 # For Precision
            if desired == 1:
                true_pos += 1 # For Recall
        else:
            if desired == 1:
                false_neg += 1 # For Recall

    print(f"Total Checking Inputs: {len(checking_data)}")
    print(f"Total True Positive: {true_pos}")
    print(f"Total False Negitive: {false_neg}")
    print(f"Total Correct Predictions: {correct_predictions}")
    print(f"Model Recall: {0 if (true_pos + false_neg) == 0 else round(((true_pos/(true_pos + false_neg)) * 100), 2)}%")
    print(f"Model Precision: {round(((correct_predictions/len(checking_data)) * 100), 2)}%")
    print("=" * 50)

if __name__ == "__main__":
    # Create data and do or not shuffle
    dataset, input_size = get_dataset(os.path.dirname(__file__) + "/data.txt", shuffle=True)
    # Create training data
    training_data, checking_data = split_data(dataset, train_percentage=0.8)

    # Create and train perceptron
    p = Perceptron(size=input_size, learning_rate=0.0000000001)

    train_perceptron(p, training_data, epochs=10000, print_step_details=False)
    print("Checking the model on the new data")
    check_perceptron(p, checking_data)
    print("Checking the model on the trained data")
    check_perceptron(p, training_data)

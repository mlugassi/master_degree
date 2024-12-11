import csv
import random
import numpy as np
import os
import tarfile
import urllib.request


def shuffle_dataset(x, y):
    cmobied_dataset = list(zip(x, y))
    random.shuffle(cmobied_dataset)
    x_randomized, y_randomized = zip(*cmobied_dataset)
    return np.array(x_randomized), np.array(y_randomized)

def split_ex1_data(dataset, train_percentage=0.8):
    if not 0 <= train_percentage <= 1:
        raise ValueError("train_percentage must be between 0 and 1.")

    # Calculate the number of items for training and testing
    total = len(dataset)
    train_count = round(total * train_percentage)

    # Split the data
    training_data = dataset[:train_count]
    testing_data = dataset[train_count:]
    return training_data, testing_data

def load_ex1_data(file_path, shuffle):
    x_dataset = list()
    y_dataset = list()
    # Open the file and read the content
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        # Loop through each row
        for row in reader:
            x_dataset.append((np.array([int(num) for num in row[:-1]])))
            y_dataset.append(np.array(0 if row[-1] == "g1" else 1))
    x_dataset, y_dataset = shuffle_dataset(x_dataset, y_dataset)
    x_train, x_test = split_ex1_data(np.array(x_dataset))
    y_train, y_test = split_ex1_data(np.array(y_dataset))
    return (x_train, y_train), (x_test, y_test)

def download_and_extract_cifar10(data_dir="cifar10_data"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    file_name = os.path.join(data_dir, "cifar-10-binary.tar.gz")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download CIFAR-10 dataset
    if not os.path.exists(file_name):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, file_name)
        print("Download complete!")

    # Extract the dataset
    if not os.path.exists(os.path.join(data_dir, "cifar-10-batches-bin")):
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete!")

def load_cifar10_batch(file_path, normalize):
    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    num_samples = data.shape[0] // 3073
    labels = data[0::3073]  # Extract labels

    reshaped_data = data.reshape(num_samples, 3073)
    images = reshaped_data[:, 1:].reshape(num_samples, 3, 32, 32)  # Image data
    images = preprocess_data(images, normalize)
    labels = np.array(labels, dtype=int)

    return images, labels

def preprocess_data(images, normalize):
    if normalize:
        images = images / 255.0  # Normalize pixel values to [0, 1]
    return images.reshape(images.shape[0], -1)  # Flatten images

def get_dataset(data_type, batch_size, normalize):
    if data_type == "ex2":
        download_and_extract_cifar10(os.path.dirname(__file__) + "/cifar10_data")
        data_dir = os.path.dirname(__file__) + "/cifar10_data/cifar-10-batches-bin"
        train_images = None
        train_labels = None
        for batch_file in os.listdir(data_dir):
            if batch_file.startswith("data_"):
                images, labels = load_cifar10_batch(f"{data_dir}/{batch_file}", normalize)
                train_images = images if train_images is None else np.vstack([train_images, images])
                train_labels = labels if train_labels is None else np.concatenate([train_labels, labels])
                train_images, train_labels = shuffle_dataset(train_images, train_labels)

        test_images, test_labels = load_cifar10_batch(f"{data_dir}/test_batch.bin", normalize)
        return split_to_batch(train_images, train_labels, batch_size), (test_images, test_labels)
    
    elif data_type == "ex1":
        data_dir = os.path.dirname(__file__)
        training_data, testing_data = load_ex1_data(data_dir + "/data.txt")
        return split_to_batch(training_data[0], training_data[1], batch_size), testing_data

def split_to_batch(x_train, y_train, batch_size=32):
    num_of_samples = x_train.shape[0] 
    num_batches = int(np.ceil(num_of_samples/batch_size))
    return np.array_split(x_train, num_batches), np.array_split(y_train, num_batches)

def calc_accuracy(y_pred, y_true):
    correct_predictions = 0
    for prediction, expected in zip(y_pred, y_true):
        if expected == np.argmax(prediction):
            correct_predictions += 1
    return round(correct_predictions/len(y_true) * 100 ,2)


class SimpleNN: 
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, actiovation="relu"):
        self.activation = actiovation

        self.weights1 = np.random.randn(input_size, hidden_size1) * 0.01 
        self.bias1 = np.zeros((1, hidden_size1)) 
        self.weights2 = np.random.randn(hidden_size1, hidden_size2) * 0.01 
        self.bias2 = np.zeros((1, hidden_size2)) 
        self.weights3 = np.random.randn(hidden_size2, hidden_size3) * 0.01 
        self.bias3 = np.zeros((1, hidden_size3))
        self.weights4 = np.random.randn(hidden_size3, output_size) * 0.01 
        self.bias4 = np.zeros((1, output_size)) 
 
    def forward(self, x): 
        self.z1 = np.dot(x, self.weights1) + self.bias1 
        self.a1 = self.activate(self.z1)
 
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2 
        self.a2 = self.activate(self.z2)
        
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.activate(self.z3)
        
        self.z4 = np.dot(self.a3, self.weights4) + self.bias4
        return self.softmax(self.z4) 
 
    def backward(self, x, y_true, y_pred, learning_rate=0.01, derivative="relu"): 
        num_samples = x.shape[0] 
        
        # Gradient for output layer 
        dz4 = y_pred 
        dz4[range(num_samples), y_true] -= 1 
        dz4 /= num_samples
 
        dw4 = np.dot(self.a3.T, dz4)
        db4 = np.sum(dz4, axis=0, keepdims=True)

        # Gradient for third hidden layer
        da3 = np.dot(dz4, self.weights4.T)
        dz3 = da3 * self.derivative(self.z3)

        dw3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Gradient for second hidden layer
        da2 = np.dot(dz3, self.weights3.T)
        dz2 = da2 * self.derivative(self.z2)
 
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Gradient for first hidden layer
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self.derivative(self.z1)
 
        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights3 -= learning_rate * dw3
        self.bias3 -= learning_rate * db3
        self.weights4 -= learning_rate * dw4
        self.bias4 -= learning_rate * db4
    
    def activate(self, z):
        if self.activation.lower() == "relu":
            return np.maximum(0, z)  
        elif self.activation.lower() == "sigmoid":
            return self.sigmoid(z)

    def compute_loss(self, y_pred, y_true): 
        num_samples = y_true.shape[0] 
        epsilon = 1e-12
        y_pred = y_pred + epsilon  # for stability
        correct_log_probs = -np.log(y_pred[range(num_samples), y_true]) 
        return np.sum(correct_log_probs) / num_samples 

    def softmax(self, x): 
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True) 
 
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, z):
        if self.activation == "relu":
            return (z > 0)
        elif self.activation == "sigmoid":
            sig = self.sigmoid(z)
            return sig * (1 - sig)

def run_model(data_type, output_size, epochs, batch_size, normalize, actiovation, learning_rate, print_epochs=False):
    train_dataset, test_dataset = get_dataset(data_type, batch_size, normalize)
    
    model = SimpleNN(input_size=train_dataset[0][0].shape[1], hidden_size1=32 * 32 * 2, hidden_size2=32 * 32 * 1, hidden_size3=32, output_size=output_size, actiovation=actiovation)
    
    # TRAIN PHASE
    print(f"---------- TRAIN PHASE - START ----------")
    for epoch in range(epochs):
        for i, (x_train, y_train) in enumerate(zip(train_dataset[0], train_dataset[1])):
            predictions = model.forward(x_train)
            model.backward(x_train, y_train, predictions.copy(), learning_rate=learning_rate)
            train_loss = model.compute_loss(predictions.copy(), y_train)
            train_acc = calc_accuracy(predictions, y_train)
            if print_epochs:
                print(f"epoch {epoch + 1} - {i + 1}/{len(train_dataset[0])} - acccurency {train_acc}%, loss {train_loss}")
    print(f"---------- TRAIN PHASE - END ----------")

    print(f"---------- TEST PHASE - START ----------")
    predictions = model.forward(test_dataset[0])
    print(f"---------- TEST PHASE - END ----------")    
    print(f"---------------------------------------")
    print("Model Setup:")
    print(f"---------------------------------------")
    print(f"Data Type: {data_type}")
    print(f"Batch Size: {batch_size}")
    print(f"Normalize: {normalize}")
    print(f"Actiovation: {actiovation}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"---------------------------------------")
    print("Model Results:")
    print(f"---------------------------------------")           
    print(f"Model Acccurency: {calc_accuracy(predictions, test_dataset[1])}%")
    print(f"Train Acccurency: {train_acc}%")
    print(f"Train Loss: {train_loss}")
    print(f"---------------------------------------")  

if __name__ == "__main__":
   run_model(data_type="ex2", output_size=10, epochs=10, batch_size=32, normalize=True, actiovation="relu",learning_rate=0.1, print_epochs=True)        
   run_model(data_type="ex2", output_size=10,  epochs=10, batch_size=32, normalize=False, actiovation="relu", learning_rate=0.1)        
   run_model(data_type="ex2", output_size=10,  epochs=10, batch_size=32, normalize=True, actiovation="sigmoid", learning_rate=0.01)        
   run_model(data_type="ex1", output_size=2,  epochs=10, batch_size=32, normalize=True, actiovation="relu", learning_rate=0.1)        
   run_model(data_type="ex1", output_size=2,  epochs=10, batch_size=32, normalize=True, actiovation="sigmoid", learning_rate=0.1)        


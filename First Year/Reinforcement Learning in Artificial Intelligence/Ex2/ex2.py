import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load the CSV file
def load_csv(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Step 2: Preprocess the data
def preprocess_data(data, selected_columns, target_column, categorical_cols=[], numerical_cols=[]):
    # Handle missing values (if any)
    x = len(data)
    data = data.dropna()
    print("DIFF:" + str(x - len(data)))
    x = data[selected_columns]
    y = data[target_column]

    # One-hot encode categorical features
    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)
    x = x.astype(int)
    
    # Normalize numerical columns
    scaler = StandardScaler()
    x[numerical_cols] = scaler.fit_transform(x[numerical_cols])

    y = y.map({"satisfied": 1, "neutral or dissatisfied": 0})
    print(y[0])
    y = tf.keras.utils.to_categorical(y)
    print(y[0])

    x, y = shuffle(x, y)
    
    return x.values, y

# Main Execution Flow
if __name__ == "__main__":
    # Define your CSV file path and target column
    train_file_path = "train.csv"
    test_file_path = "test.csv"

    # Load the data
    train_dataset = load_csv(os.path.dirname(__file__) + "/" + train_file_path)
    test_dataset = load_csv(os.path.dirname(__file__) + "/" + test_file_path)
    if train_dataset is not None and test_dataset is not None:
        # Preprocess the data
        x_train, y_train = preprocess_data(train_dataset,
                               selected_columns=["Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"],
                               target_column="satisfaction",
                               categorical_cols=["Gender", "Customer Type", "Type of Travel", "Class"],
                               numerical_cols=["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"])

        x_test, y_test = preprocess_data(test_dataset,
                               selected_columns=["Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"],
                               target_column="satisfaction",
                               categorical_cols=["Gender", "Customer Type", "Type of Travel", "Class"],
                               numerical_cols=["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"])


        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )

        # Build the model
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        # model.fit(x_train, y_train, epochs=10)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_split=0.05, callbacks=[early_stopping], epochs=10000)

        predictions = model.predict(x_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')
        predictions = model.predict(x_train)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_train, axis=1)
        accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
        print(f'Train Accuracy: {accuracy * 100:.2f}%')

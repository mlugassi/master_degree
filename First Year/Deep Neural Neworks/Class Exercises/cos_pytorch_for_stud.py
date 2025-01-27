import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate dataset
dataset = np.cos(np.arange(10000) * (20 * np.pi / 1000))[:, None]
plt.plot(dataset)
# plt.show()


# Create dataset with lookback
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# LSTM Model definition
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        predictions = self.linear(lstm_out)
        return predictions


# Data preparation
look_back = 40
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Convert to PyTorch tensors
trainX = torch.FloatTensor(trainX).reshape(-1, look_back, 1)
trainY = torch.FloatTensor(trainY)
testX = torch.FloatTensor(testX).reshape(-1, look_back, 1)
testY = torch.FloatTensor(testY)

# Initialize model, loss function, and optimizer
model = LSTMPredictor(input_dim=1, hidden_dim=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
epochs = 100
batch_size = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(trainX), batch_size):
        batch_X = trainX[i:i + batch_size]
        batch_y = trainY[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(trainX):.4f}')

# Make predictions
model.eval()
look_ahead = 500
predictions = np.zeros((look_ahead, 1))
current_input = trainX[-1].unsqueeze(0)

with torch.no_grad():
    for i in range(look_ahead):
        prediction = model(current_input).numpy()
        predictions[i] = prediction
        # Update input sequence
        new_input = torch.cat((current_input[:, 1:, :],
                               torch.FloatTensor([prediction]).reshape(1, 1, 1)),
                              dim=1)
        current_input = new_input

# Plotting
plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
plt.plot(np.arange(look_ahead), dataset[train_size:(train_size + look_ahead)], label="test function")
plt.legend()
plt.show()
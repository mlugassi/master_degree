import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
from datetime import datetime
import time
seconds = time.time()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logfile = sys.stdout
# logfile = open(f"gameNetwork_{seconds}.log", "w")

class GameNetwork(nn.Module):
    def __init__(self, board_size, device=None):
        super(GameNetwork, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}", file=logfile)
        input_dim = (board_size ** 2) * 2 + 1
        num_actions = (board_size ** 2) * 3

        # Shared layers of the network
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # The value is a scalar
            nn.Sigmoid()  # Value range is [0, 1]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),  # Action probabilities
            nn.Softmax(dim=-1)  # Normalized probabilities
        )

    def forward(self, x):
        # Compute shared layers
        shared_output = self.shared_layers(x)

        # Compute value and policy outputs
        value = self.value_head(shared_output)
        policy = self.policy_head(shared_output)

        return value, policy

    def save_weights(self, file_path):
        """Saves the network weights to a file"""
        torch.save(self.state_dict(), file_path)

    def load_weights(self, file_path, train=True):
        """Loads saved weights from a file"""
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        if train:
            self.train()
        else:
            self.eval()


class GameDataset(Dataset):
    def __init__(self, data):
        self.samples = []

        for line in data:
            x = torch.tensor(line["state"], dtype=torch.float)
            if type(line["move"]) is type(dict()):
                y_policy = torch.zeros(75, dtype=torch.float)
                for move, prob in line["move"].items():  
                    y_policy[move] = prob
                am_i_win = line["winner"]
            else:
                y_policy = F.one_hot(torch.tensor(line["move"], dtype=torch.long), num_classes=75).float()
                am_i_win = 1 if line["winner"] == line["player"] else 0

            y_value = torch.tensor(am_i_win, dtype=torch.float)
            
            self.samples.append((x, y_value, y_policy))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def load_data(json_path):
    games = []
    with open(json_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            for key in data:
                games.append(data[key])
            # games.append(data)

    return games


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), shuffle=False)

    return train_data, val_data, test_data


def train(model, train_loader, val_loader, epochs, lr):
    model.to(model.device)  # Ensure model is on GPU/CPU
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_value_fn = nn.MSELoss()
    # loss_policy_fn = nn.CrossEntropyLoss()
    loss_policy_fn = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y_value, y_policy in train_loader:
            # Move inputs and targets to the GPU if available
            x, y_value, y_policy = x.to(model.device), y_value.to(model.device), y_policy.to(model.device)

            optimizer.zero_grad()
            value_pred, policy_pred = model(x)

            # Compute losses
            loss_value = loss_value_fn(value_pred.squeeze(), y_value)
            # loss_policy = loss_policy_fn(policy_pred, y_policy)
            loss_policy = loss_policy_fn(F.log_softmax(policy_pred, dim=1), y_policy)

            loss = loss_value + loss_policy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate the model
        if val_loader:
            avg_val_loss, val_policy_acc, val_value_acc = evaluate(model, val_loader)
            print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, "
                f"Policy Accuracy: {val_policy_acc:.2%}, Value Accuracy: {val_value_acc:.2%}", file=logfile)

def evaluate(model, dataloader):
    model.eval()
    loss_value_fn = nn.MSELoss()
    loss_policy_fn = nn.CrossEntropyLoss()

    total_loss = 0
    correct_policy_predictions = 0
    total_value_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for x, y_value, y_policy in dataloader:
            # Move inputs and targets to the GPU if available
            x, y_value, y_policy = x.to(model.device), y_value.to(model.device), y_policy.to(model.device)

            value_pred, policy_pred = model(x)

            # Compute losses
            loss_value = loss_value_fn(value_pred.squeeze(), y_value)
            loss_policy = loss_policy_fn(policy_pred, y_policy)
            loss = loss_value + loss_policy
            total_loss += loss.item()

            # Policy accuracy
            predicted_moves = torch.argmax(policy_pred, dim=1)
            true_moves = torch.argmax(y_policy, dim=1)
            correct_policy_predictions += (predicted_moves == true_moves).sum().item()

            # Value accuracy
            batch_value_accuracy = 1 - torch.abs(y_value - value_pred.squeeze())
            total_value_accuracy += batch_value_accuracy.sum().item()

            total_samples += y_value.size(0)

    avg_loss = total_loss / len(dataloader)
    policy_accuracy = correct_policy_predictions / total_samples if total_samples > 0 else 0
    value_accuracy = total_value_accuracy / total_samples if total_samples > 0 else 0

    return avg_loss, policy_accuracy, value_accuracy


if __name__ == "__main__":
    board_size = 5
    num_of_iteration = 300
    batch_size = 512
    epochs = 100
    learning_rate = 0.001
    version = "2"
    network_weights_name = f"game_network_weights_{board_size}_batch_{batch_size}_v{version}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    

    json_path = f"play_book_{board_size}_v{version.split('.')[0]}.json"
    data = load_data(json_path)
    train_data, val_data, test_data = split_data(data)

    train_dataset = GameDataset(train_data)
    val_dataset = GameDataset(val_data)
    test_dataset = GameDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = GameNetwork(board_size, device)
    
    if os.path.isfile(network_weights_name):
        model.load_weights(network_weights_name, train=True)

    for i in range(num_of_iteration):
        print(f"#################### ITERATION #{i + 1} ####################", file=logfile)
        train(model, train_loader, val_loader, epochs=epochs, lr=learning_rate)
        model.save_weights(network_weights_name)
        if i % 10 == 0 and os.path.isfile(logfile.name):
            f_name = logfile.name
            logfile.close()
            logfile = open(f_name, "a")


    train_avg_loss, train_policy_accuracy, train_value_accuracy = evaluate(model, train_loader)
    test_avg_loss, test_policy_accuracy, test_value_accuracy = evaluate(model, test_loader)

    print("#################### MODEL CONFIGURATION ####################", file=logfile)
    print(f"Version: v{version}", file=logfile)
    print(f"Device: {device}", file=logfile)
    print(f"Board Size: {board_size}", file=logfile)
    print(f"Batch Size: {batch_size}", file=logfile)
    print(f"Iteration: {num_of_iteration}", file=logfile)
    print(f"Epochs: {epochs}", file=logfile)
    print(f"Learning Rate: {learning_rate}", file=logfile)
    print("#################### TRAIN RESULT ####################", file=logfile)
    print(f"Average Loss: {train_avg_loss:.6f}", file=logfile)
    print(f"Policy Accuracy: {train_policy_accuracy:.2%}", file=logfile)
    print(f"Value Accuracy: {train_value_accuracy:.2%}", file=logfile)
    print("#################### TEST RESULT ####################", file=logfile)
    print(f"Average Loss: {test_avg_loss:.6f}", file=logfile)
    print(f"Policy Accuracy: {test_policy_accuracy:.2%}", file=logfile)
    print(f"Value Accuracy: {test_value_accuracy:.2%}", file=logfile)
    print("#####################################################", file=logfile)

    end_time = datetime.now()
    print(f"Training completed at: {end_time}")
    print(f"Total training time: {end_time - start_time}")
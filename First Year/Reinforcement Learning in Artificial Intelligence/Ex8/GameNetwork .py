import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class GameNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(GameNetwork, self).__init__()
        # encode board
        # player
        # 

        # Shared layers of the network
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
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
        self.load_state_dict(torch.load(file_path))
        if train:
            self.train()
        else:
            self.eval()  # Switch to evaluation mode

class GameDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        
        for line in data:
            x = torch.tensor(line["state"] , dtype=torch.float)
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

def train(model, dataloader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_value_fn = nn.MSELoss()  # הפסד עבור חיזוי ערך המשחק
    loss_policy_fn = nn.CrossEntropyLoss()  # הפסד עבור חיזוי המהלך

    for epoch in range(epochs):
        total_loss = 0
        for x, y_value, y_policy in dataloader:
            optimizer.zero_grad()
            
            value_pred, policy_pred = model(x)

            # חישוב הפסדים
            loss_value = loss_value_fn(value_pred.squeeze(), y_value)
            loss_policy = loss_policy_fn(policy_pred, y_policy)

            loss = loss_value + loss_policy
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Example usage of the network
if __name__ == "__main__":
    board_size = 5
    input_dim = (board_size**2) *2 + 1  # Example: number of features to represent the board
    num_actions = (board_size**2) * 3  # Example: number of possible actions in the game
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


    # Create the network
    model = GameNetwork(input_dim, num_actions)
    
    # Load weights
    if os.path.isfile(f"game_network_weights_{board_size}.pth"):
        model.load_weights(f"game_network_weights_{board_size}.pth", train=True)

    # קריאה לאימון
    json_path = f"play_book_{board_size}.json"  # נתיב הקובץ שלך
    data = load_data(json_path)
    dataset = GameDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train(model, dataloader, epochs=1*10, lr=0.001)

    # Create random input (for demonstration purposes)
    # sample_input = torch.randn((1, input_dim))  # Input for a single example
   
    # state = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        #    , "move": 21, "player": 1, "winner": 1}}

    # state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                #, "move": 51, "player": 1, "winner": 1},
    
    state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
            #  "move": 54, "player": 1, "winner": 1},
    
    #  "move": 51, "player": -1, "winner": -1}}
    #state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1]
#                  "move": 48, "player": -1, "winner": -1}}
    sample_input = torch.tensor(state, dtype=torch.float)
    
    # "move": 52, "player": -1, "winner": -1}


    # Predict value and policy probabilities
    value, policy = model(sample_input)

    print("######## MODEL RESULTS ##########")
    print(f"Predicted Value: {value.item():.6f}")
    print("Move Idex:", torch.argmax(policy).item())
    print("Move Probability:", torch.max(policy, dim=0).values.item())
    # print("Predicted Policy Probabilities:", policy.detach().numpy())
    print("Predicted Policy Probabilities:")
    policy_np = policy.detach().numpy().flatten()  # Convert to NumPy array and flatten (if needed)
    for idx, prob in enumerate(policy_np):
        print(f"#{idx}: {prob:.15f}")
    print("################################")

    # Save weights
    model.save_weights(f"game_network_weights_{board_size}.pth")



    

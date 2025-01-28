import torch
import torch.nn as nn
import torch.optim as optim

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
            nn.Tanh()  # Value range is [-1, 1]
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

    def load_weights(self, file_path):
        """Loads saved weights from a file"""
        self.load_state_dict(torch.load(file_path))
        self.eval()  # Switch to evaluation mode

# Example usage of the network
if __name__ == "__main__":
    input_dim = 42  # Example: number of features to represent the board
    num_actions = 7  # Example: number of possible actions in the game

    # Create the network
    network = GameNetwork(input_dim, num_actions)

    # Create random input (for demonstration purposes)
    sample_input = torch.randn((1, input_dim))  # Input for a single example

    # Predict value and policy probabilities
    value, policy = network(sample_input)

    print("Predicted Value:", value.item())
    print("Predicted Policy Probabilities:", policy.detach().numpy())

    # Save weights
    network.save_weights("game_network_weights.pth")

    # Load weights
    network.load_weights("game_network_weights.pth")

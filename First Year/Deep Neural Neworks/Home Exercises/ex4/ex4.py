
import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import os
import shutil

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pathlib import Path

from clearml import Task 

task = Task.init(project_name="Deep Neural Neworks Ex4", task_name="DataSet Version #1") 

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=2, img_dim=(300,300)):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)  # Input: 3x300x300, Output: 32X100X100
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=0) # Output: 64x?X?
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) # Output: 128x?X?
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=3, padding=0) # Output: 64x?X?
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1) # Output: 32x?X?
        
        # Pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)  # Halves spatial dimensions
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 32)  # Adjusted for image size (7x7 after pooling)
        self.fc2 = nn.Linear(32, num_classes)  # Output layer with `num_classes` units

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))  # Apply conv1, ReLU, and pool
        x = self.pool2(torch.relu(self.conv2(x)))  # Apply conv2, ReLU, and pool
        x = self.pool1(torch.relu(self.conv3(x)))  # Apply conv3, ReLU, and pool
        x = self.pool2(torch.relu(self.conv4(x)))  # Apply conv4, ReLU, and pool
        x = torch.relu(self.conv5(x))  # Apply conv5 and ReLU (no pooling here)
        
        x = torch.flatten(x, 1)  # Flatten the output from 3D (32x7x7) to 1D (32*7*7)
        x = torch.relu(self.fc1(x))  # Apply fully connected layer 1
        x = self.fc2(x)  # Apply fully connected layer 2 (output layer)
        return x

# Step 1: Extract the zip file
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def organize_images(src_dir, dest_dir):
    smoking_dir = os.path.join(dest_dir, 'smoking')
    not_smoking_dir = os.path.join(dest_dir, 'notsmoking')
    
    # Create directories if they don't exist
    os.makedirs(smoking_dir, exist_ok=True)
    os.makedirs(not_smoking_dir, exist_ok=True)

    # Move images based on your naming convention or manually (if needed)
    for img_name in os.listdir(src_dir):
        if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.jpeg') or img_name.lower().endswith('.png'):
            if img_name.startswith('smoking'):
                shutil.move(os.path.join(src_dir, img_name), os.path.join(smoking_dir, img_name))
            else:
                shutil.move(os.path.join(src_dir, img_name), os.path.join(not_smoking_dir, img_name))
    
    print(f"Images organized into {smoking_dir} and {not_smoking_dir}")    

# Step 2: Create a DataLoader for the dataset
def prepare_data(data_dir, batch_size=32):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Resize all images to 300x300 (can be adjusted based on input size)
        transforms.ToTensor(),  # Convert image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for RGB
    ])

    
    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.classes

# Step 3: Training loop
def train_model(model, dataloader, num_epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# Step 4: Full pipeline
if __name__ == "__main__":
    zip_path = "First Year\Deep Neural Neworks\Home Exercises\ex4\student_305536575.zip"  # Path to the zip file
    extract_to = "First Year\Deep Neural Neworks\Home Exercises\ex4\student_305536575"       # Folder where the zip will be extracted
    data_folder_path = Path(extract_to, "Training\Training\smoking")
    batch_size = 1

    # Extract dataset
    if not os.path.exists(extract_to):
        extract_zip(zip_path, extract_to)
        organize_images(data_folder_path, data_folder_path)
    
    # Prepare data
    dataloader, class_names = prepare_data(data_dir=Path(extract_to, "Training\Training\smoking"), batch_size=batch_size)
    print(f"Classes: {class_names}")
    
    # Define and train the model
    model = CNN(num_classes=len(class_names))
    train_model(model, dataloader, num_epochs=5)

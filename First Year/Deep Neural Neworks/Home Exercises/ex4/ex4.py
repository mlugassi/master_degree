import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import os
import shutil

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from pathlib import Path

from clearml import Task 

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=2, img_dim=(300, 300)):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=3, padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 32)  # Adjusted for image size (7x7 after pooling)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool1(torch.relu(self.conv3(x)))
        x = self.pool2(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 1: Extract the zip file
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

def organize_images(src_dir, dest_dir):
    smoking_dir = os.path.join(dest_dir, 'smoking')
    not_smoking_dir = os.path.join(dest_dir, 'notsmoking')
    
    os.makedirs(smoking_dir, exist_ok=True)
    os.makedirs(not_smoking_dir, exist_ok=True)

    for img_name in os.listdir(src_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            if img_name.startswith('smoking'):
                shutil.move(os.path.join(src_dir, img_name), os.path.join(smoking_dir, img_name))
            else:
                shutil.move(os.path.join(src_dir, img_name), os.path.join(not_smoking_dir, img_name))
    
    print(f"Images organized into {smoking_dir} and {not_smoking_dir}")

# Step 2: Create DataLoaders for the dataset
def prepare_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    

    return dataloader, dataset.classes

# Step 3: Training loop with validation
def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
      
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

def test_model(data_type, model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    print(f"Data Tested: {data_type} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")              

# Step 4: Full pipeline
if __name__ == "__main__":
    root_path = "First Year\Deep Neural Neworks\Home Exercises\ex4"
    task_name = "DataSet_V1"
    model_path = Path(root_path, task_name + ".model")
    zip_path = Path(root_path, "student_305536575.zip")
    extract_to =Path(root_path, "student_305536575")
    train_data_folder_path = Path(extract_to, "Training\Training\smoking")
    validation_data_folder_path = Path(extract_to, "Validation\Validation\smoking")
    test_data_folder_path = Path(extract_to, "Testing\Testing\smoking")
    batch_size = 32

    # task = Task.init(project_name="Deep Neural Neworks Ex4", task_name=task_name) 

    if not os.path.exists(extract_to):
        extract_zip(zip_path, extract_to)
        organize_images(train_data_folder_path, train_data_folder_path)
        organize_images(validation_data_folder_path, validation_data_folder_path)
        organize_images(test_data_folder_path, test_data_folder_path)
    train_loader, class_names = prepare_data(data_dir=train_data_folder_path, batch_size=batch_size)
    val_loader, _ = prepare_data(data_dir=validation_data_folder_path, batch_size=batch_size)
    test_loader, _ = prepare_data(data_dir=test_data_folder_path, batch_size=batch_size)

    model = CNN(num_classes=len(class_names))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from '{model_path}'")
        
    train_model(model, train_loader, val_loader, num_epochs=10)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at '{model_path}'")

    # Test the model
    test_model("Trained Data", model, train_loader)
    test_model("New Data", model, test_loader)

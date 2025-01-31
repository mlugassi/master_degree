import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import os
import shutil

from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from pathlib import Path

from clearml import Task 

os.environ['http_proxy'] = 'http://proxy-iil.intel.com:912'
os.environ['https_proxy'] = 'http://proxy-iil.intel.com:912'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# Define ResNet model
class ResNetModel(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

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

def prepare_data(data_dir, batch_size=32, img_resize=(300, 300), use_augmentation=False):
    # Create a custom dataset that includes image paths
    class ImageFolderWithPaths(datasets.ImageFolder):
        def __getitem__(self, index):
            # Get the image and its label
            img, label = super().__getitem__(index)
            # Get the image path
            img_path = self.imgs[index][0]
            return img, label, img_path

    train_transform = [
        transforms.Resize(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if use_augmentation:
        train_transform.insert(1, transforms.RandomHorizontalFlip())
        train_transform.insert(2, transforms.RandomRotation(15))
        train_transform.insert(3, transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    transform = transforms.Compose(train_transform)    
    dataset = ImageFolderWithPaths(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, dataset.classes

# Step 3: Training loop with validation
def train_model(logger, model, train_loader, val_loader, optimizer, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels, _ in train_loader:
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
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
      
        logger.report_scalar("Loss", "Train", iteration=epoch, value=train_loss)
        logger.report_scalar("Loss", "Validation", iteration=epoch, value=val_loss)
        logger.report_scalar("Accuracy", "Validation", iteration=epoch, value=val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

def test_model(data_type, model, test_loader, data_folder_path, task_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    misclassified_dir = Path(data_folder_path, f"misclassified_samples_{task_name}_{data_type}/")
    if misclassified_dir.exists():
        shutil.rmtree(misclassified_dir)
    misclassified_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for inputs, labels, img_paths in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
                # Save misclassified images
            for i in range(len(inputs)):
                if predicted[i] != labels[i]:  
                    original_path = img_paths[i]
                    predicted_label = predicted[i].item()
                    filename = Path(original_path).name
                    new_filename = f"its_{test_loader.dataset.classes[labels[i]]}_but_pred_{test_loader.dataset.classes[predicted_label]}_img_{filename}"
                    shutil.copy(original_path, misclassified_dir / new_filename)

    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    print(f"{data_type} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")              

# Step 4: Full pipeline
if __name__ == "__main__":
    # Confiuration 
    task_name = "DatasetQ2V1"

    batch_size = 10
    epochs = 50
    learning_rate = 0.001
    img_resize = (50, 50)
    use_regulation = False
    use_augmentation = False
    load_last_model = False
    do_training = True
    save_model = True
    test_train_data = True
    test_test_data = False
    pretrained = False
    optimizer = "Adam"
    # optimizer = "RMSprop"

    # Define and log parameters
    params = {
        "task_name": task_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "img_resize": img_resize,
        "use_regulation": use_regulation,
        "use_augmentation": use_augmentation,
        "load_last_model": load_last_model,
        "do_training": do_training,
        "save_model": save_model,
        "test_train_data": test_train_data,
        "test_test_data": test_test_data,
        "pretrained": pretrained,
        "optimizer": optimizer
    }

    root_path = "First Year\Deep Neural Neworks\Home Exercises\ex4"
    model_path = Path(root_path, task_name + ".model")
    if task_name == "DatasetQ2V1":
        zip_path = Path(root_path, "student_305536575.zip")
        extract_to =Path(root_path, "student_305536575")
    else:
        zip_path = Path(root_path, "student_305536575_v2.zip")
        extract_to =Path(root_path, "student_305536575_v2")        
    train_data_folder_path = Path(extract_to, "Training\Training\smoking")
    validation_data_folder_path = Path(extract_to, "Validation\Validation\smoking")
    test_data_folder_path = Path(extract_to, "Testing\Testing\smoking")

    task = Task.init(project_name="DNN Ex4", task_name=task_name) 
    logger = task.get_logger()
    task.connect(params)  # Log parameters to Configuration
    
    if not os.path.exists(extract_to):
        extract_zip(zip_path, extract_to)
        organize_images(train_data_folder_path, train_data_folder_path)
        organize_images(validation_data_folder_path, validation_data_folder_path)
        organize_images(test_data_folder_path, test_data_folder_path)

    train_loader, class_names = prepare_data(data_dir=train_data_folder_path, batch_size=batch_size, img_resize=img_resize, use_augmentation=use_augmentation)
    val_loader, _ = prepare_data(data_dir=validation_data_folder_path, batch_size=batch_size, img_resize=img_resize, use_augmentation=False)
    test_loader, _ = prepare_data(data_dir=test_data_folder_path, batch_size=batch_size, img_resize=img_resize, use_augmentation=False)

    model = ResNetModel(len(class_names), pretrained=pretrained)

    if load_last_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from '{model_path}'")

    if do_training:
        if optimizer == "RMSprop":
            opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-3 if use_regulation else 0)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3 if use_regulation else 0)

        train_model(logger, model, train_loader, val_loader, num_epochs=epochs, optimizer=opt)

    if save_model:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at '{model_path}'")

    # Final Summary Print
    print("\n===== FINAL SUMMARY =====")
    print(f"Task Name: {task_name}")
    print("Configuration:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    print("=========================\n")
    print("\n===== TEST RESULT =====")
    test_model("VALIDATION DATA", model, val_loader, root_path, task_name)
    if test_train_data:
        test_model("TRAINED DATA", model, train_loader, root_path, task_name)
    if test_test_data:
        test_model("TEST DATA", model, test_loader, root_path, task_name)
    print("=========================\n")

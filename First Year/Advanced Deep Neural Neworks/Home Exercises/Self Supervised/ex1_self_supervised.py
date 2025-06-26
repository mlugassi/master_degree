import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import argparse
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as F

class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices=None):
        self.base_data = base_dataset if indices is None else Subset(base_dataset, indices)
        self.rotation_angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.base_data) * 4

    def __getitem__(self, idx):
        base_idx = idx // 4
        rotation_idx = idx % 4
        image, _ = self.base_data[base_idx]
        angle = self.rotation_angles[rotation_idx]
        rotated_image = F.rotate(image, angle)
        return rotated_image, rotation_idx

def get_first_n_per_class(dataset, samples_per_class=60):
    class_to_indices = {i: [] for i in range(10)}
    selected_indices = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if len(class_to_indices[label]) < samples_per_class:
            class_to_indices[label].append(idx)
            selected_indices.append(idx)

        if all(len(v) == samples_per_class for v in class_to_indices.values()):
            break

    return selected_indices

def train(model, loader, device, num_epochs, criterion, optimizer, freeze_backbone=False):
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small', action='store_true', help='Train on 600 deterministic images using supervised learning')
    parser.add_argument('--supervised_small', action='store_true', help='Self-supervised rotation pretraining on full dataset + fine-tune on 600 deterministic labeled images')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training (default: 500)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation (default: 128)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    base_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    if args.supervised_small:
        indices = get_first_n_per_class(base_train_dataset, samples_per_class=60)

        print("Phase 1: Self-supervised training on rotation task (full dataset)")
        rotation_dataset = RotationDataset(base_train_dataset)
        rotation_loader = DataLoader(rotation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 4)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train(model, rotation_loader, device, args.num_epochs, criterion, optimizer)

        print("Phase 2: Fine-tuning on real CIFAR-10 labels (600 deterministic samples)")
        backbone = model

        model = resnet18(pretrained=False)
        model.load_state_dict(backbone.state_dict())
        model.fc = nn.Linear(model.fc.in_features, 10)

        fine_tune_dataset = Subset(base_train_dataset, indices)
        fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train(model, fine_tune_loader, device, args.num_epochs, criterion, optimizer, freeze_backbone=True)

        print("Evaluation on test set:")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        evaluate(model, test_loader, device)

    elif args.small:
        indices = get_first_n_per_class(base_train_dataset, samples_per_class=60)
        train_dataset = Subset(base_train_dataset, indices)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Training on 600 deterministic supervised images")
        train(model, train_loader, device, args.num_epochs, criterion, optimizer)

        print("Evaluation on test set:")
        evaluate(model, test_loader, device)

    else:
        train_loader = DataLoader(base_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Training on full CIFAR-10 dataset")
        train(model, train_loader, device, args.num_epochs, criterion, optimizer)

        print("Evaluation on test set:")
        evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()

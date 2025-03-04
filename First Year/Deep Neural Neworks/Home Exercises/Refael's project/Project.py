import torch
import torchvision
import json
import os
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

# Custom dataset for LabelMe annotations
class ScrollDataset(Dataset):
    def __init__(self, json_dir, img_dir, transforms=None):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.transforms = transforms
        self.files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        json_path = os.path.join(self.json_dir, self.files[idx])
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_path = os.path.join(self.img_dir, data["imagePath"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        for shape in data["shapes"]:
            x_min = min([p[0] for p in shape["points"]])
            y_min = min([p[1] for p in shape["points"]])
            x_max = max([p[0] for p in shape["points"]])
            y_max = max([p[1] for p in shape["points"]])
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # Class label 1 for scroll segment
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        image = F.to_tensor(image)
        
        return image, target
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Load dataset
    json_dir = "./train"  # Path to LabelMe JSON files
    img_dir = "./train"  # Path to images
    dataset = ScrollDataset(json_dir, img_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Load pre-trained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    n_classes = 2  # Background + scroll segment
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), "faster_rcnn.pth")
    print("Model saved!")

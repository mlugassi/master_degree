import torch
import torchvision
import json
import os
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import numpy as np
import torchvision.transforms as T
from PIL import Image
import glob


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
    

# פונקציה להערכת המודל על סט הבדיקה
def evaluate_model(model, test_images_path, labelme_annotations_path, iou_threshold=0.5):
    model.eval()  # מצב הערכה (ללא גרדיאנטים)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = T.Compose([T.ToTensor()])
    total_iou = []
    
    # טעינת התמונות
    image_files = glob.glob(os.path.join(test_images_path, "*.jpg"))
    
    for img_path in image_files:
        img_name = os.path.basename(img_path).replace(".jpg", ".json")
        annotation_path = os.path.join(labelme_annotations_path, img_name)

        if not os.path.exists(annotation_path):
            print(f"Missing annotation for {img_path}, skipping.")
            continue

        # טעינת תמונה
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # הפעלת המודל
        with torch.no_grad():
            prediction = model(image_tensor)

        predicted_boxes = prediction[0]['boxes'].cpu().numpy()
        predicted_scores = prediction[0]['scores'].cpu().numpy()

        # טעינת תיבות ייחוס מ-LabelMe
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        
        ground_truth_boxes = [shape["points"] for shape in annotation["shapes"]]
        ground_truth_boxes = [[x1, y1, x2, y2] for [[x1, y1], [x2, y2]] in ground_truth_boxes]

        # חישוב ה-IOU לכל תיבה
        image_ious = []
        for gt_box in ground_truth_boxes:
            best_iou = 0
            for pred_box, score in zip(predicted_boxes, predicted_scores):
                if score < 0.5:  # סינון ניבויים עם ביטחון נמוך
                    continue
                iou = calculate_iou(pred_box, gt_box)
                best_iou = max(best_iou, iou)
            image_ious.append(best_iou)

        avg_iou = np.mean(image_ious) if image_ious else 0
        total_iou.append(avg_iou)
        print(f"Image: {img_path}, Average IOU: {avg_iou:.4f}")

    # תוצאה סופית
    overall_iou = np.mean(total_iou) if total_iou else 0
    print(f"\nFinal Model IOU Score: {overall_iou:.4f}")

    return overall_iou

# פונקציה לחישוב IOU בין שתי תיבות חוסמות
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # חישוב חיתוך
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # חישוב איחוד
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Load dataset
    json_dir = "train"  # Path to LabelMe JSON files
    img_dir = "train"  # Path to images
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

    # טוען את המודל שאומן
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 2  # רקע + מגילה
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load("faster_rcnn.pth"))  # טוען מודל מאומן

    # הערכת המודל על סט הבדיקה
    evaluate_model(model, "train", "train")


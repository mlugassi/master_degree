import torch
import torchvision
import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torchvision.transforms as T
from PIL import Image
import glob
import csv
from clearml import Task
from datetime import datetime
task_name = f"Project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
task = Task.init(project_name="DNN2", task_name=task_name)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

class Models:
    faster_rcnn     = "faster_rcnn"
    faster_rcnn_1   = "faster_rcnn.1.pth"
    faster_rcnn_2   = "faster_rcnn.2.pth"
    faster_rcnn_3   = "faster_rcnn.3.pth"
    faster_rcnn_4   = "faster_rcnn.4.pth"
    faster_rcnn_5   = "faster_rcnn.5.pth"
    faster_rcnn_6   = "faster_rcnn.6.pth"
    faster_rcnn_7   = "faster_rcnn.7.pth"
    faster_rcnn_8   = "faster_rcnn.8.pth"
    faster_rcnn_9   = "faster_rcnn.9.pth"
    retinanet       = "retinanet"
    retinanet_1     = "retinanet.1.pth"
    retinanet_2     = "retinanet.2.pth"
    retinanet_3     = "retinanet.3.pth"
    retinanet_4     = "retinanet.4.pth"
    retinanet_5     = "retinanet.5.pth"
    retinanet_7     = "retinanet.7.pth"
    retinanet_8     = "retinanet.8.pth"
    retinanet_9     = "retinanet.9.pth"

def detect_bounding_boxes(model, image_path: str, threshold=0.6):
    model.eval()  # מצב הערכה (ללא גרדיאנטים)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    # טעינת תמונה
    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # הפעלת המודל
    with torch.no_grad():
        predictions = model(image_tensor)
    
    boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
    scores = predictions[0]['scores'].cpu().numpy()
    mask = scores > threshold
    fixed_bboxes = boxes[mask]

    return sorted(fixed_bboxes, key=lambda box: (box[0]**2 + box[1]**2)**0.5)

def create_fasterrcnn_model(pretrained):
    
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    return model

def create_retinanet_model(pretrained, num_classes=2):
    from torchvision.models.detection import retinanet_resnet50_fpn
    model = retinanet_resnet50_fpn(pretrained_backbone=pretrained,  num_classes=num_classes)

    return model

def predict_process_bounding_boxes(image_path: str, output_csv: str, model_name=Models.faster_rcnn_1) -> None:
    """
    מזהה תיבות חוסמות בתמונה, ממספר אותן ושומר את התוצאות ב-CSV.
    """
    pretrained = False
    if model_name.startswith(Models.faster_rcnn):
        model = create_fasterrcnn_model(pretrained)
    elif model_name.startswith(Models.retinanet):
        model = create_retinanet_model(pretrained)
    else:
        print("-E- Model_name:", model_name ,"not founded.")
    
    model.load_state_dict(torch.load(model_name, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    boxes = detect_bounding_boxes(model, image_path)
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "scroll_number", "xmin", "ymin", "xmax", "ymax", "iou"])
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            writer.writerow([os.path.basename(image_path), i+1, xmin, ymin, xmax, ymax, -1])

def draw_bounding_boxes(image_path, csv_path, output_path):
    """
    Draws bounding boxes on the image based on detection results from a CSV file.
    
    Args:
        image_path (str): Path to the image file.
        csv_path (str): Path to the CSV file containing bounding box information.
        output_path (str): Path to save the output image with bounding boxes drawn.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            _, scroll_number, xmin, ymin, xmax, ymax, iou = row
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            label = f"Scroll {scroll_number}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Saved image with bounding boxes at: {output_path}")

def boxes_from_csv(csv_path):
    boxes = []
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            # _, _, xmin, ymin, xmax, ymax, _ = row
            xmin, ymin, xmax, ymax = map(int, row[2:6])
            boxes.append([xmin,ymin,xmax,ymax])
    return boxes

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

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
        #image = F.to_tensor(image)

        image = torch.tensor(image).permute(2, 0, 1)  # HWC -> CHW

        return image, target
    
# פונקציה להערכת המודל על סט הבדיקה
def evaluate_model(test_images_path, labelme_annotations_path, results = "./results", model_name = None, last_epoch=False):
    csv_dir_res = f"{results}/csv"
    os.makedirs(csv_dir_res, exist_ok=True)
    model_name = model_name.replace("./","")

    if last_epoch:
        img_dir_res = f"{results}/images"
        os.makedirs(img_dir_res, exist_ok=True)

    total_iou = []
    # טעינת התמונות
    image_files = glob.glob(os.path.join(test_images_path, "*.jpg"))
    
    for img_path in image_files:
        # if "M43991-1-E.jpg" not in img_path: continue

        json_name = os.path.basename(img_path).replace(".jpg", ".json")
        annotation_path = os.path.join(labelme_annotations_path, json_name)

        img_name = os.path.basename(img_path)
        csv_path = f"{csv_dir_res}/" + img_name.replace(".jpg",f".{model_name}.csv")
        img_output_path = None
        
        if last_epoch:
            img_output_path = f"{img_dir_res}/" + img_name.replace(".jpg",f".{model_name}.jpg")
        elif not os.path.exists(annotation_path):
            continue
        iou = evaluate_image(img_path, annotation_path, img_output_path, csv_path, model_name)
        if iou:
            total_iou.append(iou)

    # תוצאה סופית
    overall_iou = np.mean(total_iou) if total_iou else 0
    print(f"\nFinal Model IOU Score - {test_images_path}: {overall_iou:.4f}")

    return overall_iou

def evaluate_image(image_path, annotation_path, image_with_boxes=None, csv_path=None, model_name=None):

    # if not os.path.exists(annotation_path):
    #     # print(f"Missing annotation for {img_path}, skipping.")
    #     return None

    predict_process_bounding_boxes(image_path, csv_path, model_name=model_name)
    predicted_boxes = boxes_from_csv(csv_path)
    if image_with_boxes is not None:
        draw_bounding_boxes(image_path,csv_path,image_with_boxes)
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    ground_truth_boxes = [shape["points"] for shape in annotation["shapes"]]
    ground_truth_boxes = [[x1, y1, x2, y2] for [[x1, y1], [x2, y2]] in ground_truth_boxes]

    # חישוב ה-IOU לכל תיבה
    image_ious = []
    for gt_box in ground_truth_boxes:
        best_iou = 0
        for pred_box in predicted_boxes:
            iou = calculate_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        image_ious.append(best_iou)

    avg_iou = np.mean(image_ious) if image_ious else 0
    print(f"Image: {image_path}, Average IOU: {avg_iou:.4f}")
    return avg_iou

if __name__ == "__main__":
    # inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_dir = "train_augmented"  # Path to LabelMe JSON files
    img_dir = "train_augmented"  # Path to images
    num_classes = 2  # Background + scroll segment

    # batch = 2
    # model_name = Models.retinanet
    # lr = 0.0001
    # num_epochs = 10*1000

    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_1, 2, 0.0001, 0*1000, True  # --> 15321
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_2, 1, 0.00001, 3*1000, True  # --> 15333
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_2, 1, 0.00001, 3*1000, False  # --> 15333
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_3, 2, 0.0001,  3*1000, False # --> 15326
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_4, 2, 0.001,  3*1000, True  # --> 15324
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_5, 2, 0.0001, 6*1000, True  # --> 15325
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_6, 1, 0.0001, 10*1000, True  # --> 15336

    model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_7, 2, 0.0001, 1*800, True  # --> 15497
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_8, 1, 0.0001, 1*800, True  # --> 15481
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_9, 2, 0.0001, 2*800, True  # --> 15483
    
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_1,   1, 0.0001, 3*1000, True  # --> 15328
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_2,   2, 0.0001, 3*1000, True  # --> 15327
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_3,   2, 0.0001, 3*1000, False # --> 15329
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_4,   2, 0.001,  3*1000, True  # --> 15330
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_5,   2, 0.0001, 6*1000, True  # --> 15331

    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_7, 2, 0.0001, 1*800, True  # --> 15476
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_8, 1, 0.0001, 1*1000, True  # --> 15473
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_9, 2, 0.0001, 2*1000, True  # --> 15472

    dataset = ScrollDataset(json_dir, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    if os.path.exists(model_name) or pretrained is False:
        pretrained = False
    else:
        pretrained = True
    
    params = {
        "batch": batch,
        "epochs": num_epochs,
        "learning_rate": lr,
        "model_name": model_name,
        "pretrained": pretrained,
        "device": str(device)
    }
    task.connect(params)
    logger = task.get_logger()

    print("###########################################")
    for key in params:
        print(f"# {key}: {params[key]}")
    print("###########################################")

    if model_name.startswith(Models.faster_rcnn):
        model = create_fasterrcnn_model(pretrained)
    elif model_name.startswith(Models.retinanet):
        model = create_retinanet_model(pretrained, num_classes)
    else:
        print("Error: Not found model_name:", model_name)
        exit(1)
    
    # Load pre-trained
    if pretrained is False and os.path.isfile(model_name):
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
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
        logger.report_scalar(title="Loss", series="Train", value=loss.item(), iteration=epoch+1)

        # Save trained model
        if epoch and epoch % 100 == 0:
            torch.save(model.state_dict(), model_name)
            print(f"Model {model_name} saved!")

            with torch.no_grad():
                iou_train = evaluate_model(model_name=model_name, test_images_path="train_augmented",  labelme_annotations_path="train_augmented")
                iou_test  = evaluate_model(model_name=model_name, test_images_path="test_augmented",   labelme_annotations_path="test_augmented")
                logger.report_scalar(title="IOU", series="Train", value=iou_train, iteration=epoch)
                logger.report_scalar(title="IOU", series="Test",  value=iou_test,  iteration=epoch)
            model.train()
    
    if num_epochs:
        torch.save(model.state_dict(), model_name)
        print(f"Model {model_name} saved!")
    
    with torch.no_grad():
        iou_train = evaluate_model(model_name=model_name, test_images_path="train_augmented",  labelme_annotations_path="train_augmented", last_epoch=True)
        iou_test  = evaluate_model(model_name=model_name, test_images_path="test_augmented",   labelme_annotations_path="test_augmented",  last_epoch=True)
        if num_epochs:
            logger.report_scalar(title="IOU", series="Train", value=iou_train, iteration=epoch)
            logger.report_scalar(title="IOU", series="Test",  value=iou_test,  iteration=epoch)


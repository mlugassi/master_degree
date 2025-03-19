import torch
import torchvision
import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import numpy as np
import torchvision.transforms as T
from PIL import Image
import glob
import torch
import torchvision
import gzip
import shutil
import re
import csv
from clearml import Task
from datetime import datetime
task_name = f"Project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
task = Task.init(project_name="DNN", task_name=task_name)


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    return obj

def extract_gz(gz_file, output_file=None):
    if not gz_file.endswith(".gz"):
        raise ValueError("Not a .gz file")

    if output_file is None:
        output_file = gz_file[:-3]  # מסיר את הסיומת .gz
    with gzip.open(gz_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Extracted: {gz_file} -> {output_file}")
    return output_file

def compress_to_gz(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + ".gz"  # מוסיף סיומת .gz אם לא סופק שם פלט

    with open(input_file, 'rb') as f_in, gzip.open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Compressed: {input_file} -> {output_file}")
    return output_file

def split_and_compress_pth(input_file, max_size=50 * 1024 * 1024):  # גודל מרבי לכל חלק (כאן 100MB)
    part_num = 0
    with open(input_file, 'rb') as f_in:
        while True:
            chunk = f_in.read(max_size)
            if not chunk:
                break
            part_num += 1
            part_filename = f"{input_file}.part{part_num}.gz"
            with gzip.open(part_filename, 'wb') as f_out:
                f_out.write(chunk)
            print(f"Created: {part_filename}")

def merge_and_decompress_pth(input_file):
    # חיפוש כל הקבצים עם התבנית המתאימה
    part_files = sorted(glob.glob(f"{input_file}.part*.gz"), key=lambda x: int(re.search(r'\d+', x).group()))
    
    if not part_files:
        print("No parts found!")
        return
    
    with open(input_file, 'wb') as f_out:
        for part_filename in part_files:
            with gzip.open(part_filename, 'rb') as f_in:
                f_out.write(f_in.read())
            print(f"Merged: {part_filename}")

    print(f"Reconstructed file saved as {input_file}")

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

def load_model(model_path: str):
    """טוען את מודל ה- RetinaNet המאומן."""
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

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
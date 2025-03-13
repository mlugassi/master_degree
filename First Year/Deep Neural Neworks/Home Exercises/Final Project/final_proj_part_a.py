import torch
import random
import json
import cv2
import numpy as np
import os
import csv
import shutil
from ultralytics import YOLO

# os.environ['http_proxy'] = 'http://proxy-iil.intel.com:912'
# os.environ['https_proxy'] = 'http://proxy-iil.intel.com:912'
# os.environ['no_proxy'] = 'localhost,127.0.0.1'

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_annotations_from_image(dir_path, image_name, labels_to_num: dict, print_warning=True):
    """Gets bounding box annotations from a JSON file that has the same name as the image."""
    annotation_path = os.path.join(dir_path, "labels", image_name.split('.')[0] + ".json")
    img_path = os.path.join(dir_path, "images", image_name)
    if print_warning and not os.path.exists(annotation_path):
        print(f"Warning: No annotation file found for {annotation_path}")
        return []
    
    with open(annotation_path, 'r') as file:
        data = json.load(file)
    
    annotations = []
    scroll_counter = 1
   
    image = cv2.imread(img_path)
    image_height, image_width = image.shape[:2]
    
    for shape in sorted(data['shapes'], key=lambda s: distance(s['points'][0], s['points'][1])):  # Sort by Euclidean distance
        if shape['shape_type'] == 'rectangle':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            annotations.append((labels_to_num[shape['label']], x_center / image_width, y_center / image_height, width / image_width, height / image_height, scroll_counter))
            scroll_counter += 1
    return annotations

def save_annotations_yolo(annotations, output_txt):
    """Saves annotations in YOLO format (class_id x_center y_center width height)."""
    with open(output_txt, 'w') as file:
        for label, x_center, y_center, width, height, scroll_number in annotations:
            file.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def train_yolo(dataset_yaml, trained_model_path, epochs=10, learning_rate=0.001, optimizer='Adam', create_test_model=False):
    """Trains YOLOv8 using images from the specified directory and evaluates accuracy."""
    model_path = trained_model_path if os.path.exists(trained_model_path) else os.path.join(os.path.dirname(trained_model_path), "yolov8n.pt")
    model = YOLO(model_path)
    model.train(data=dataset_yaml, epochs=epochs, lr0=learning_rate, optimizer=optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.val()
    model.save(trained_model_path)
    if create_test_model:
        model.save("test_model.pt")


def prepare_data(input_dir, output_dir, labels_to_num: dict, train_precent=0.9):
    input_train_dir = os.path.join(input_dir, 'train', 'images')        
    input_test_dir = os.path.join(input_dir, 'test', 'images')      
    if not os.path.exists(output_dir):
        train_labels_dir = os.path.join(output_dir, 'train', 'labels')
        train_images_dir = os.path.join(output_dir, 'train', 'images')
        val_images_dir = os.path.join(output_dir, 'val', 'images')
        val_labels_dir = os.path.join(output_dir, 'val', 'labels')
        test_images_dir = os.path.join(output_dir, 'test', 'images')
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        os.makedirs(test_images_dir, exist_ok=True)
        
        
        input_train_images = [f for f in os.listdir(input_train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        input_test_images = [f for f in os.listdir(input_test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not input_train_images:
            raise FileNotFoundError("No images found in train directory. Ensure images are correctly placed.")

        if not input_test_images:
            raise FileNotFoundError("No images found in train directory. Ensure images are correctly placed.")
                
        random.shuffle(input_train_images)
        split_idx = int(len(input_train_images) * train_precent)
        train_images = input_train_images[:split_idx]
        val_images = input_train_images[split_idx:]
        
        annotations = {img: get_annotations_from_image(os.path.join(input_dir, 'train') , img, labels_to_num) for img in input_train_images}
        
        for image_name in train_images:
            label_path = os.path.join(train_labels_dir, os.path.splitext(image_name)[0] + '.txt')
            save_annotations_yolo(annotations[image_name], label_path)
            shutil.copy2(os.path.join(input_train_dir, image_name), os.path.join(train_images_dir, image_name))
        
        for image_name in val_images:
            label_path = os.path.join(val_labels_dir, os.path.splitext(image_name)[0] + '.txt')
            save_annotations_yolo(annotations[image_name], label_path)
            shutil.copy2(os.path.join(input_train_dir, image_name), os.path.join(val_images_dir, image_name))
        
        for image_name in input_test_images:
            shutil.copy2(os.path.join(input_test_dir, image_name), os.path.join(test_images_dir, image_name))

        dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
        output_dir_path = os.path.abspath(output_dir).replace('\\', '/')
        lables = [("\"" + k + "\"") for k in labels_to_num.keys()]
        yaml_content = f"path: {output_dir_path}\ntrain: train/images\nval: val/images\n\nnc: {len(labels_to_num)}\nnames: [{', '.join(lables)}]"
            
        with open(dataset_yaml, 'w') as f:
            f.write(yaml_content)
    else:
        input_train_images = [f for f in os.listdir(input_train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        input_test_images = [f for f in os.listdir(input_test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]        

    return os.path.join(output_dir, 'dataset.yaml'), input_train_images, input_test_images

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


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (tuple): (xmin, ymin, xmax, ymax) for first bounding box.
        box2 (tuple): (xmin, ymin, xmax, ymax) for second bounding box.
    
    Returns:
        float: IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0


def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    
    Args:
    image_path (str): Path to the input image. (full paths to the input images)
    output_csv (str): Path to the output CSV file.
    """

    model = YOLO("test_model.pt")  # Load YOLO model
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    results = model.predict(image, augment=True, agnostic_nms=True)
    bounding_boxes = []
    
    detected_boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            detected_boxes.append((x1, y1, x2, y2, confidence))
    
    # Sort by Euclidean distance from the top-left corner
    detected_boxes.sort(key=lambda b: np.sqrt(b[0]**2 + b[1]**2))
    
    label_path = os.path.join(image_path.lstrip(os.sep).split(os.sep)[0], image_path.lstrip(os.sep).split(os.sep)[1], "labels") + os.path.basename(image_path).split('.')[0] + ".txt"
    scroll_counter = 1
    for x1, y1, x2, y2, confidence in detected_boxes:
        iou = calculate_iou((x1, y1, x2, y2), (0, 0, image.shape[1], image.shape[0])) if os.path.exists(label_path) else -1
        bounding_boxes.append((os.path.basename(image_path), scroll_counter, int(x1), int(y1), int(x2), int(y2), round(iou, 2)))
        scroll_counter += 1
    
    # Save bounding boxes to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'scroll_number', 'xmin', 'ymin', 'xmax', 'ymax', 'iou'])
        writer.writerows(bounding_boxes)
    
    print(f"Bounding boxes saved to {output_csv}")

# Example usage
labels_to_num = {'scroll': 0}
epochs = 1
learning_rate=0.001
optimizer='Adam'
output_dir = "YOLOv8"
val_precent = 0.9
version = 1
model_output = os.path.join(output_dir, 'yolov8_trained_v' + str(version) + '.pt')
input_model = model_output
train_model = True
test_model = True
draw_boxes = True

dataset_yaml, train_images, test_images = prepare_data(input_dir="./", output_dir=output_dir, labels_to_num=labels_to_num, train_precent=val_precent)
if train_model:
    train_yolo(dataset_yaml=dataset_yaml, trained_model_path=model_output, epochs=epochs, learning_rate=learning_rate, optimizer=optimizer, create_test_model=test_model)

if test_model:
    train_prediction_path = os.path.join(output_dir, "prediction_train")
    os.makedirs(train_prediction_path, exist_ok=True)

    for img in train_images:
        image_path = os.path.join(output_dir, "train", "images", img)
        csv_path = os.path.join(train_prediction_path, img.split('.')[0] + ".csv")
        pred_output_img_path = os.path.join(output_dir, "prediction_train", "predicted_" + img)

        predict_process_bounding_boxes(image_path, csv_path)
        if draw_boxes:
            draw_bounding_boxes(image_path, csv_path, pred_output_img_path)
import torch
import random
import json
import cv2
import numpy as np
import os
import shutil
from ultralytics import YOLO

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_annotations_from_image(dir_path, image_name, labels_to_num: dict):
    """Loads bounding box annotations from a JSON file that has the same name as the image."""
    annotation_path = os.path.join(dir_path, "labels", image_name.split('.')[0] + ".json")
    img_path = os.path.join(dir_path, "images", image_name)
    if not os.path.exists(annotation_path):
        print(f"Warning: No annotation file found for {annotation_path}")
        return []
    
    with open(annotation_path, 'r') as file:
        data = json.load(file)
    
    annotations = []
    scroll_counter = 1
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
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

def train_yolo(input_dir, output_dir, model_output, labels_to_num: dict, val_precent=0.1, epochs=10, learning_rate=0.001, optimizer='Adam'):
    """Trains YOLOv8 using images from the specified directory and evaluates accuracy."""
    if not os.path.exists(output_dir):
        labels_dir = os.path.join(input_dir, 'labels')
        images_dir = os.path.join(input_dir, 'images')        
        train_labels_dir = os.path.join(output_dir, 'train', 'labels')
        train_images_dir = os.path.join(output_dir, 'train', 'images')
        val_images_dir = os.path.join(output_dir, 'val', 'images')
        val_labels_dir = os.path.join(output_dir, 'val', 'labels')
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise FileNotFoundError("No images found in train directory. Ensure images are correctly placed.")
        
        random.shuffle(image_files)
        split_idx = int(len(image_files) * val_precent)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        annotations = {img: load_annotations_from_image(input_dir, img, labels_to_num) for img in image_files}
        
        for image_name in train_images:
            label_path = os.path.join(train_labels_dir, os.path.splitext(image_name)[0] + '.txt')
            save_annotations_yolo(annotations[image_name], label_path)
            shutil.copy2(os.path.join(images_dir, image_name), os.path.join(train_images_dir, image_name))
        
        for image_name in val_images:
            label_path = os.path.join(val_labels_dir, os.path.splitext(image_name)[0] + '.txt')
            save_annotations_yolo(annotations[image_name], label_path)
            shutil.copy2(os.path.join(images_dir, image_name), os.path.join(val_images_dir, image_name))
        
        dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
        output_dir_path = os.path.abspath(output_dir).replace('\\', '/')
        lables = [("\"" + k + "\"") for k in labels_to_num.keys()]
        yaml_content = f"path: {output_dir_path}\ntrain: train/images\nval: val/images\n\nnc: {len(labels_to_num)}\nnames: [{', '.join(lables)}]"
        print(yaml_content)
            
        with open(dataset_yaml, 'w') as f:
            f.write(yaml_content)
        print(f"Dataset YAML successfully created with {len(image_files)} images and {len(os.listdir(train_labels_dir))} annotations:")
    else:
        dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
    # Print dataset.yaml content for debugging (only once)
    
    model = YOLO(os.path.join(output_dir, 'yolov8n.pt'))
    model.train(data=dataset_yaml, epochs=epochs, lr0=learning_rate, optimizer=optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.val()
    model.save(os.path.join(output_dir, model_output))

# Example usage
train_dir = "./train"
test_dir = "./test"
model_output = "yolov8_trained.pt"
labels_to_num = {'scroll': 0}

# Train YOLO with images from specified 'train' directory
train_yolo(train_dir, "YOLOv8", model_output, labels_to_num)

import torch
import random
import json
import cv2
import numpy as np
import os
import csv
import shutil
from ultralytics import YOLO
from clearml import Task, Logger
import glob


os.environ['http_proxy'] = 'http://proxy-iil.intel.com:912'
os.environ['https_proxy'] = 'http://proxy-iil.intel.com:912'
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def calculate_mean_iou(csv_folder):
    """
    Reads all CSV files in a folder, extracts the 'iou' column, and calculates the mean IoU.
    
    Args:
        csv_folder (str): Path to the folder containing CSV files.
    
    Returns:
        float: The mean IoU across all files, or None if no valid values found.
    """
    iou_values = []

    for file_name in os.listdir(csv_folder):
        if file_name.endswith(".csv"):
            csv_path = os.path.join(csv_folder, file_name)
            with open(csv_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Read the header
                
                if header and "iou" in header:
                    iou_index = header.index("iou")
                    for row in reader:
                        try:
                            iou_value = float(row[iou_index])
                            if iou_value >= 0:
                                iou_values.append(iou_value)
                        except ValueError:
                            continue  # Skip non-numeric values

    if iou_values:
        mean_iou = sum(iou_values) / len(iou_values)
        return mean_iou
    else:
        return None  # No valid IoU values found
    
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_annotations_from_image(dir_path, image_name, labels_to_num: dict, print_warning=True):
    """Gets bounding box annotations from a JSON file that has the same name as the image."""
    annotation_path = os.path.join(dir_path, "labels", image_name.split('.')[0] + ".json")
    img_path = os.path.join(dir_path, "images", image_name)
    if not os.path.exists(annotation_path):
        if print_warning:
            print(f"Warning: No annotation file found for {annotation_path}")
        return []
    
    with open(annotation_path, 'r') as file:
        data = json.load(file)
    
    annotations = []
    scroll_counter = 1
   
    image = cv2.imread(img_path)
    image_height, image_width = image.shape[:2]
    
    boxes = list()
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            x_0, y_0 = shape['points'][0]
            x_1, y_1 = shape['points'][1]        
            boxes.append((min(x_0, x_1), min(y_0, y_1), max(x_0, x_1), max(y_0, y_1), labels_to_num[shape['label']]))

    for box in sorted(boxes, key=lambda b: distance((0,0), (b[0], b[1]))):  # Sort by Euclidean distance
            annotations.append((box[4], box[0] / image_width, box[1] / image_height, box[2] / image_width, box[3] / image_height, scroll_counter))
            scroll_counter += 1
    return annotations

def save_annotations_yolo(annotations, output_txt):
    """Saves annotations in YOLO format (class_id x_center y_center width height)."""
    with open(output_txt, 'w') as file:
        for label, x_center, y_center, width, height, scroll_number in annotations:
            file.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def train_yolo(dataset_yaml, trained_model_path, epochs, learning_rate, optimizer, batch_size, mosaic_augmentation, image_size, pretrained, create_test_model):
    """Trains YOLOv8 using images from the specified directory and evaluates accuracy."""
    model_path = trained_model_path if os.path.exists(trained_model_path) else os.path.join(os.path.dirname(trained_model_path), "yolov8n.pt")
    model = YOLO(model_path)

    print("\n🔎 Running training...")
    # Train the model
    model.train(data=dataset_yaml, epochs=epochs, lr0=learning_rate, optimizer=optimizer, 
                batch=batch_size, mosaic=mosaic_augmentation, imgsz=image_size, pretrained=pretrained,
                device='cuda' if torch.cuda.is_available() else 'cpu')
    val_results = model.val()
    model.save(trained_model_path)
    if create_test_model:
        model.save("best_model_q1.pt")

    return val_results

def prepare_data(input_dir, output_dir, labels_to_num: dict, train_precent=0.9):
    input_train_dir = os.path.join(input_dir, 'train', 'images')        
    input_test_dir = os.path.join(input_dir, 'test', 'images')      
    if not os.path.exists(output_dir):
        train_labels_dir = os.path.join(output_dir, 'train', 'labels')
        train_images_dir = os.path.join(output_dir, 'train', 'images')
        val_images_dir = os.path.join(output_dir, 'val', 'images')
        val_labels_dir = os.path.join(output_dir, 'val', 'labels')
        test_images_dir = os.path.join(output_dir, 'test', 'images')
        test_labels_dir = os.path.join(output_dir, 'test', 'labels')
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        os.makedirs(test_images_dir, exist_ok=True)
        os.makedirs(test_labels_dir, exist_ok=True)
        
        
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

        train_annotations = {img: get_annotations_from_image(os.path.join(input_dir, 'train') , img, labels_to_num) for img in input_train_images}
        test_annotations = {img: get_annotations_from_image(os.path.join(input_dir, 'test') , img, labels_to_num, print_warning=False) for img in input_test_images}
        
        for image_name in train_images:
            label_path = os.path.join(train_labels_dir, os.path.splitext(image_name)[0] + '.txt')
            save_annotations_yolo(train_annotations[image_name], label_path)
            shutil.copy2(os.path.join(input_train_dir, image_name), os.path.join(train_images_dir, image_name))
        
        for image_name in val_images:
            label_path = os.path.join(val_labels_dir, os.path.splitext(image_name)[0] + '.txt')
            save_annotations_yolo(train_annotations[image_name], label_path)
            shutil.copy2(os.path.join(input_train_dir, image_name), os.path.join(val_images_dir, image_name))
        
        for image_name in input_test_images:
            label_path = os.path.join(test_labels_dir, os.path.splitext(image_name)[0] + '.txt')
            save_annotations_yolo(test_annotations[image_name], label_path)            
            shutil.copy2(os.path.join(input_test_dir, image_name), os.path.join(test_images_dir, image_name))

        dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
        output_dir_path = os.path.abspath(output_dir).replace('\\', '/')
        lables = [("\"" + k + "\"") for k in labels_to_num.keys()]
        yaml_content = f"path: {output_dir_path}\ntrain: train/images\nval: val/images\n\nnc: {len(labels_to_num)}\nnames: [{', '.join(lables)}]"
            
        with open(dataset_yaml, 'w') as f:
            f.write(yaml_content)

    return os.path.join(output_dir, 'dataset.yaml')

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
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_iou_results(output_dir, images_type, draw_boxes):
    prediction_path = os.path.join(output_dir, ("prediction_" + images_type))
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path, exist_ok=True)
        for img in [f for f in os.listdir(os.path.join(output_dir, images_type, "images")) if f.endswith(('.png', '.jpg', '.jpeg'))]:
            image_path = os.path.join(output_dir, images_type, "images", img)
            csv_path = os.path.join(prediction_path, img.split('.')[0] + ".csv")
            predict_process_bounding_boxes(image_path, csv_path)
            if draw_boxes:
                draw_imgs_dir_path = os.path.join(prediction_path, "images")
                os.makedirs(draw_imgs_dir_path, exist_ok=True)
                draw_bounding_boxes(image_path, csv_path, os.path.join(draw_imgs_dir_path, "predicted_" + img))    
    return calculate_mean_iou(os.path.join(output_dir, ("prediction_" + images_type)))

def get_label_boxes(label_path, image_width, image_height):
    """
    Loads normalized YOLO labels and converts them to pixel coordinates.

    Args:
        label_path (str): Path to the label file.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        list: List of bounding boxes in pixel coordinates [(xmin, ymin, xmax, ymax)].
    """
    if not os.path.exists(label_path):
        return []

    boxes = dict()
    scroll_num = 1
    with open(label_path, "r") as file:
        for line in file.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_min, y_min, x_max, y_max = map(float, parts)
            x_min = int(x_min * image_width)
            y_min = int(y_min * image_height)
            x_max = int(x_max * image_width)
            y_max = int(y_max * image_height)
            boxes[scroll_num] = (x_min, y_min, x_max, y_max)
            scroll_num += 1
    return boxes

def predict_process_bounding_boxes(image_path: str, output_csv: str) -> None:
    """
    Processes an image to detect bounding boxes around scroll segments.
    Saves the bounding box data to a CSV file.
    
    Args:
    image_path (str): Path to the input image. (full paths to the input images)
    output_csv (str): Path to the output CSV file.
    """

    model = YOLO("best_model_q1.pt")  # Load YOLO model
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

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
            x_min, y_min, x_max, y_max, confidence, class_id = box.tolist()
            detected_boxes.append((x_min, y_min, x_max, y_max, confidence))
    
    # Sort by Euclidean distance from the top-left corner
    detected_boxes.sort(key=lambda b: distance((b[0], b[1]), (0, 0)))
    
    label_path = os.path.join(image_path.lstrip(os.sep).split(os.sep)[0], image_path.lstrip(os.sep).split(os.sep)[1], "labels", os.path.basename(image_path).split('.')[0] + ".txt")
    label_boxes = get_label_boxes(label_path, image_width, image_height)
    scroll_counter = 1

    for x_min, y_min, x_max, y_max, confidence in detected_boxes:
        iou = calculate_iou((x_min, y_min, x_max, y_max), label_boxes[scroll_counter]) if scroll_counter in label_boxes else -1
        bounding_boxes.append((os.path.basename(image_path), scroll_counter, int(x_min), int(y_min), int(x_max), int(y_max), round(iou, 2)))
        scroll_counter += 1
    
    # Save bounding boxes to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'scroll_number', 'xmin', 'ymin', 'xmax', 'ymax', 'iou'])
        writer.writerows(bounding_boxes)
    
    print(f"Bounding boxes saved to {output_csv}")

if __name__ == "__main__":
    labels_to_num = {'scroll': 0}

    config = {
        "epochs": 1000,
        "learning_rate": 0.001,
        "batch_size": 16,
        "pretrained": False,
        "mosaic_augmentation": 0,
        "image_size": 640,
        "optimizer": "Adam",
        "train_precent": 0.8,
        "version": 1,
        "question": 1,
        "train_model": True,
        "test_model": True,
        "draw_boxes": True,
    }
    config["output_dir"] = f"YOLO8_Q{config['question']}V{config['version']}"
    config["model_output"] = os.path.join(config["output_dir"], f"yolov8_trained_q{config['question']}_v{config['version']}.pt")
    config["input_model"] = config["model_output"]
    config["input_dir"] = f"./dataset_q{config['question']}/"

    # Print configuration details
    print("\n############# MODEL CONFIGURATION ################")
    for key, value in config.items():
        print(f"###### {key}: {value}")
    print("############################################")

    task = Task.init(project_name="DNN - Final Project", task_name=f"YOLO8_Q{config['question']}v{config['version']}") 
    task.connect(config)  # Log parameters to Configuration

    dataset_yaml = prepare_data(input_dir=config['input_dir'], output_dir=config['output_dir'], labels_to_num=labels_to_num, train_precent=config['train_precent'])
    if config['train_model']:
        model_results = train_yolo(dataset_yaml=dataset_yaml, trained_model_path=config['model_output'], 
                                   epochs=config['epochs'], learning_rate=config['learning_rate'], optimizer=config['optimizer'],
                                   batch_size=config['batch_size'], mosaic_augmentation=config['mosaic_augmentation'], image_size=config['image_size'],
                                   pretrained=config['pretrained'], create_test_model=config['test_model'])
    
    if config['test_model']:
        mean_iou_train = calculate_iou_results(output_dir=config['output_dir'], images_type="train", draw_boxes=config['draw_boxes'])
        mean_iou_test = calculate_iou_results(output_dir=config['output_dir'], images_type="test", draw_boxes=config['draw_boxes'])

    sample_image_path = os.path.abspath(os.path.join(config['input_dir'], "sample_image.jpg"))  # Change to a real image path
    sample_csv_path = os.path.abspath(os.path.join(config['output_dir'], "sample_output.csv")) # Output CSV file
    predict_process_bounding_boxes(sample_image_path, sample_csv_path) # Ensures that the predict_process_bounding_boxes function works correctly with absolute file paths.

    try:
        shutil.rmtree("runs")
        print(f"Deleted directory: runs/")
    except FileNotFoundError:
        print("Directory does not exist.")

    # Print configuration details
    print("\n############# MODEL CONFIGURATION ################")
    for key, value in config.items():
        print(f"###### {key}: {value}")
    print("############################################")
    if config['train_model']:
        print("\n############# MODEL RESULTS ################")
        print(f"##### Box Loss: {model_results.box.map50:.4f}")
        print(f"##### Precision: {model_results.results_dict['metrics/precision(B)']:.4f}")
        print(f"##### Recall: {model_results.results_dict['metrics/recall(B)']:.4f}")
        print(f"##### mAP@0.5: {model_results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"##### mAP@0.5:0.95: {model_results.results_dict['metrics/mAP50-95(B)']:.4f}")
        print(f"##### Fitness: {model_results.results_dict['fitness']:.4f}")
    if config['test_model']:
        print("\n############# TEST RESULTS ################")
        print(f"##### Mean Train IoU: {mean_iou_train if mean_iou_train is not None else 0:.4f}")
        print(f"##### Mean Test IoU: {mean_iou_test if mean_iou_test is not None else 0:.4f}")
        print("############################################")    
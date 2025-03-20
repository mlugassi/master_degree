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
from datetime import datetime


# os.environ['http_proxy'] = 'http://proxy-iil.intel.com:912'
# os.environ['https_proxy'] = 'http://proxy-iil.intel.com:912'
# os.environ['no_proxy'] = 'localhost,127.0.0.1'

# Ensure script runs from its own directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def calc_iou_information(output_dir, images_type):
    """
    Reads all CSV files in a folder, extracts the 'iou' column, and calculates the mean IoU.
    
    Args:
        output_dir (str): Path to the output directory where predictions and labels are stored.
        images_type (str): The dataset type (e.g., 'train' or 'test').

    Returns:
        float: The mean IoU across all images, including unmatched boxes as 0, or None if no valid values found.
    """
    all_matches_iou_values = []
    all_matches_unmatches_iou_values = []
    all_unmatches_act_boxes = []
    all_unmatches_pred_boxes = []

    csv_path = os.path.join(output_dir, f"prediction_{images_type}", images_type + "_prediction.csv")
    label_dir = os.path.join(output_dir, images_type, "labels")
    images_dir = os.path.join(output_dir, images_type, "images")
    num_of_images = 0
    for img in os.listdir(images_dir):
        if not img.endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(images_dir, img)
        label_path = os.path.join(label_dir, img.split('.')[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        num_of_images+=1
        # Load image to get its dimensions
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        # Load predicted boxes
        pred_boxes = []
        with open(csv_path, mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)
            if header and "xmin" in header:
                image_name_index = header.index("image_name")
                x_min_index = header.index("xmin")
                y_min_index = header.index("ymin")
                x_max_index = header.index("xmax")
                y_max_index = header.index("ymax")
                found = False
                for row in reader:
                    if row[image_name_index] == img:
                        pred_box = (int(row[x_min_index]), int(row[y_min_index]), int(row[x_max_index]), int(row[y_max_index]))
                        pred_boxes.append(pred_box)
                        found = True
                    elif found:
                        break

        # Load actual boxes
        act_boxes = []
        with open(label_path, mode='r', newline='') as label_file:
            for line in label_file:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts)
                actual_box = convert_yolo_to_bbox((x_center_norm, y_center_norm, width_norm, height_norm), image_width, image_height)
                act_boxes.append(actual_box)

        # Match predicted boxes to actual boxes and calculate IoU
        matched_iou = []
        mataches_unmatched_iou = []
        unmatched_pred = len(pred_boxes)  # Initially assume all predicted boxes are unmatched
        unmatched_act = len(act_boxes)  # Initially assume all actual boxes are unmatched

        if len(pred_boxes) > 0 and len(act_boxes) > 0:
            matched_pairs = match_boxes(pred_boxes, act_boxes)
            unmatched_pred -= len(matched_pairs)
            unmatched_act -= len(matched_pairs)

            for (pred_idx, act_idx) in matched_pairs:
                iou = calculate_iou(pred_boxes[pred_idx], act_boxes[act_idx])
                matched_iou.append(iou)
                mataches_unmatched_iou.append(iou)

        all_unmatches_pred_boxes.append(unmatched_pred)
        all_unmatches_act_boxes.append(unmatched_act)
        # Assign IoU = 0 for unmatched predicted and actual boxes
        mataches_unmatched_iou.extend([0] * unmatched_pred)  # False positives
        mataches_unmatched_iou.extend([0] * unmatched_act)  # False negatives

        if matched_iou:
            all_matches_iou_values.extend(matched_iou)
        if mataches_unmatched_iou:
            all_matches_unmatches_iou_values.extend(mataches_unmatched_iou)

    avg_matches_iou = sum(all_matches_iou_values) / len(all_matches_iou_values) if all_matches_iou_values else -1
    avg_matches_unmatched_iou = sum(all_matches_unmatches_iou_values) / len(all_matches_unmatches_iou_values) if all_matches_unmatches_iou_values else -1
    avg_unmatches_pred_boxes = sum(all_unmatches_pred_boxes) / len(all_unmatches_pred_boxes) if all_unmatches_pred_boxes else -1
    avg_unmatches_act_boxes = sum(all_unmatches_act_boxes) / len(all_unmatches_act_boxes) if all_unmatches_act_boxes else -1
    
    return num_of_images, avg_matches_iou, avg_matches_unmatched_iou, avg_unmatches_pred_boxes, avg_unmatches_act_boxes

def match_boxes(pred_boxes, act_boxes):
    """
    Matches predicted boxes to actual boxes using a greedy approach.
    The best-matching predicted box is assigned to each actual box based on IoU.

    Args:
        pred_boxes (list of tuples): List of predicted bounding boxes.
        act_boxes (list of tuples): List of actual bounding boxes.

    Returns:
        list of tuples: Matched (pred_idx, act_idx) pairs.
    """
    matched_pairs = []
    assigned_pred = set()
    assigned_act = set()

    for act_idx, act_box in enumerate(act_boxes):
        best_iou = 0
        best_pred_idx = None
        for pred_idx, pred_box in enumerate(pred_boxes):
            if pred_idx in assigned_pred:
                continue  # Skip already assigned predictions
            iou = calculate_iou(pred_box, act_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        if best_pred_idx is not None:
            matched_pairs.append((best_pred_idx, act_idx))
            assigned_pred.add(best_pred_idx)
            assigned_act.add(act_idx)

    return matched_pairs

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_annotations_from_image(dir_path, image_name, labels_to_num: dict, print_warning=True):
    """Gets bounding box annotations from a JSON file that has the same name as the image."""
    annotation_path = os.path.join(dir_path, "labels", image_name.split('.')[0] + ".json")
    img_path = os.path.join(dir_path, "images", image_name)
    if not os.path.exists(annotation_path):
        if print_warning:
            print(f"Warning: No annotation file found for {annotation_path}", flush=True)
        return None
    
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
            x1, y1 = box[0], box[1]
            x2, y2 = box[2], box[3]
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            annotations.append((box[4], x_center / image_width, y_center / image_height, width / image_width, height / image_height, scroll_counter))
            scroll_counter += 1
    return annotations

def convert_yolo_to_bbox(yolo_bbox, image_width, image_height):
    """
    Converts YOLO format bounding box (normalized) to (x_min, y_min, x_max, y_max) in absolute pixels.
    
    Args:
        yolo_bbox (tuple): (x_center_norm, y_center_norm, width_norm, height_norm) in YOLO format (normalized 0-1).
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) in absolute pixel values.
    """
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_bbox
    
    # Convert normalized values to absolute pixel coordinates
    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    width = width_norm * image_width
    height = height_norm * image_height

    # Compute x_min, y_min, x_max, y_max
    x_min = int(x_center - (width / 2))
    y_min = int(y_center - (height / 2))
    x_max = int(x_center + (width / 2))
    y_max = int(y_center + (height / 2))

    return x_min, y_min, x_max, y_max

def save_annotations_yolo(annotations, output_txt):
    """Saves annotations in YOLO format (class_id x_center y_center width height)."""
    with open(output_txt, 'w') as file:
        for label, x_center, y_center, width, height, scroll_number in annotations:
            file.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def train_yolo(dataset_yaml, trained_model_path, epochs, learning_rate, optimizer, batch_size, mosaic_augmentation, image_size, pretrained, create_test_model):
    """Trains YOLOv8 using images from the specified directory and evaluates accuracy."""
    model_path = "last_model_q2.pt" if os.path.exists("last_model_q2.pt") else os.path.join(os.path.dirname(trained_model_path), "yolov8l.pt")
    model = YOLO(model_path)

    print("\n🔎 Running training...", flush=True)
    # Train the model
    train_res = model.train(data=dataset_yaml, epochs=epochs, lr0=learning_rate, optimizer=optimizer, 
                batch=batch_size, mosaic=mosaic_augmentation, imgsz=image_size, pretrained=pretrained,
                device='cuda' if torch.cuda.is_available() else 'cpu')
    val_results = model.val()
    model.save("last_model_q2.pt")

    return train_res, val_results

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
        
        if input_train_images:
            random.shuffle(input_train_images)
            split_idx = int(len(input_train_images) * train_precent)
            train_images = input_train_images[:split_idx]
            val_images = input_train_images[split_idx:]

            train_annotations = {img: get_annotations_from_image(os.path.join(input_dir, 'train') , img, labels_to_num) for img in input_train_images}
            
            for image_name in train_images:
                label_path = os.path.join(train_labels_dir, os.path.splitext(image_name)[0] + '.txt')
                save_annotations_yolo(train_annotations[image_name], label_path)
                shutil.copy2(os.path.join(input_train_dir, image_name), os.path.join(train_images_dir, image_name))
            
            for image_name in val_images:
                label_path = os.path.join(val_labels_dir, os.path.splitext(image_name)[0] + '.txt')
                save_annotations_yolo(train_annotations[image_name], label_path)
                shutil.copy2(os.path.join(input_train_dir, image_name), os.path.join(val_images_dir, image_name))

        if input_test_images:
            test_annotations = {img: annotations for img in input_test_images if (annotations := get_annotations_from_image(os.path.join(input_dir, 'test'), img, labels_to_num, print_warning=False))}        

            for image_name in input_test_images:
                if image_name in test_annotations:
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

def draw_bounding_boxes(images_dir, csv_path, output_path):
    with open(csv_path, mode='r') as file:
        image_path = None
        reader = csv.reader(file)
        next(reader)
        
        for row in reader:
            image_name, scroll_number, xmin, ymin, xmax, ymax = row
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
            if image_path is None:
                image_path = os.path.join(images_dir, image_name)
                image = cv2.imread(image_path)                
            if image_name not in image_path:
                cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)), image)
                print(f"Saved image with bounding boxes at: {output_path}", flush=True)                
                image_path = os.path.join(images_dir, image_name)
                image = cv2.imread(image_path)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            label = f"Scroll {scroll_number}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



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

def predict_boxes(output_dir, images_type, draw_boxes):
    prediction_path = os.path.join(output_dir, ("prediction_" + images_type))
    output_csv_path = os.path.join(prediction_path, images_type + "_prediction.csv")

    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path, exist_ok=True)
        input_images_dir = os.path.join(output_dir, images_type, "images")
        images_pathes = [os.path.abspath(os.path.join(input_images_dir, f)) for f in os.listdir(input_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        process_detailed_bounding_boxes(images_pathes, output_csv_path)
        if draw_boxes:
            draw_imgs_dir_path = os.path.join(prediction_path, "images")
            os.makedirs(draw_imgs_dir_path, exist_ok=True)
            draw_bounding_boxes(input_images_dir, output_csv_path, draw_imgs_dir_path)    

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

    boxes = list()
    sorted_boxes = dict()
    with open(label_path, "r") as file:
        for line in file.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, x_center_norm, y_center_norm, widht_norm, height_norm = map(float, parts)
            boxes.append(convert_yolo_to_bbox(yolo_bbox=(x_center_norm, y_center_norm, widht_norm, height_norm), 
                                                     image_width=image_width, 
                                                     image_height=image_height))
    
    for scroll_num, box in enumerate(sorted(boxes, key=lambda b: distance((0,0), (b[0], b[1]))), start=1):
        sorted_boxes[scroll_num] = box
    
    return sorted_boxes

def process_detailed_bounding_boxes(image_paths: list[str], output_csv: str) -> None:
    model = YOLO(f"best_model_q2.pt")  # Load YOLO model

    bounding_boxes = []

    for image_path in image_paths:
        detected_boxes = []
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image at {image_path}", flush=True)
            return
        
        results = model.predict(image, augment=True, agnostic_nms=True)
        
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.data:
                x_min, y_min, x_max, y_max, confidence, class_id = box.tolist()
                detected_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        
        for scroll_num, box in enumerate(sorted(detected_boxes, key=lambda b: distance((b[0], b[1]), (0, 0))), start=1):
            bounding_boxes.append((os.path.basename(image_path), scroll_num, box[0], box[1], box[2], box[3]))
    
    # Save bounding boxes to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_name', 'scroll_number', 'xmin', 'ymin', 'xmax', 'ymax'])
        writer.writerows(bounding_boxes)

    print(f"Bounding boxes saved to {output_csv}", flush=True)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Training started at: {start_time}", flush=True)

    print("CUDA Available:", torch.cuda.is_available(), flush=True)
    print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)    
    train_res = None
    labels_to_num = {'scroll': 0, 'Megila': 0}

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
        "question": 2,
        "train_model": False,
        "test_model": True,
        "draw_boxes": True
    }
    config["output_dir"] = f"YOLO8_Q{config['question']}V{config['version']}"
    config["model_output"] = os.path.join(config["output_dir"], f"yolov8_trained_q{config['question']}_v{config['version']}.pt")
    config["input_model"] = config["model_output"]
    config["input_dir"] = f"./dataset_q{config['question']}/"

    # Print configuration details
    print("\n############# MODEL CONFIGURATION ################", flush=True)
    for key, value in config.items():
        print(f"###### {key}: {value}", flush=True)
    print("############################################", flush=True)

    task = Task.init(project_name="DNN - Final Project", task_name=f"YOLO8_Q{config['question']}v{config['version']}") 
    task.connect(config)  # Log parameters to Configuration

    dataset_yaml = prepare_data(input_dir=config['input_dir'], output_dir=config['output_dir'], labels_to_num=labels_to_num, train_precent=config['train_precent'])
    if config['train_model']:
        train_res, model_results = train_yolo(dataset_yaml=dataset_yaml, trained_model_path=config['model_output'], 
                                   epochs=config['epochs'], learning_rate=config['learning_rate'], optimizer=config['optimizer'],
                                   batch_size=config['batch_size'], mosaic_augmentation=config['mosaic_augmentation'], image_size=config['image_size'],
                                   pretrained=config['pretrained'], create_test_model=config['test_model'])
        
        shutil.move(os.path.join(train_res.save_dir, "weights", "best.pt"), f"./best_model_q{config['question']}.pt")
        predict_boxes(output_dir=config['output_dir'], images_type="train", draw_boxes=config['draw_boxes'])
        num_of_train_images, avg_train_matches_iou, avg_train_matches_unmatched_iou, avg_train_unmatches_pred_boxes, avg_train_unmatches_act_boxes = calc_iou_information(output_dir=config['output_dir'], images_type="train")
    
    if config['test_model']:
        predict_boxes(output_dir=config['output_dir'], images_type="test", draw_boxes=config['draw_boxes'])
        num_of_test_images, avg_test_matches_iou, avg_test_matches_unmatched_iou, avg_test_unmatches_pred_boxes, avg_test_unmatches_act_boxes = calc_iou_information(output_dir=config['output_dir'], images_type="test")
    
    if train_res:
        try:
            shutil.rmtree(train_res.save_dir)
            print(f"Deleted directory: {train_res.save_dir}", flush=True)
        except FileNotFoundError:
            print("Directory does not exist.", flush=True)

    # Print configuration details
    print("\n############# MODEL CONFIGURATION ################", flush=True)
    for key, value in config.items():
        print(f"###### {key}: {value}", flush=True)
    print("############################################", flush=True)
    if config['train_model']:
        print("\n############# MODEL RESULTS ################", flush=True)
        print(f"##### Box Loss: {model_results.box.map50:.4f}", flush=True)
        print(f"##### Precision: {model_results.results_dict['metrics/precision(B)']:.4f}", flush=True)
        print(f"##### Recall: {model_results.results_dict['metrics/recall(B)']:.4f}", flush=True)
        print(f"##### mAP@0.5: {model_results.results_dict['metrics/mAP50(B)']:.4f}", flush=True)
        print(f"##### mAP@0.5:0.95: {model_results.results_dict['metrics/mAP50-95(B)']:.4f}", flush=True)
        print(f"##### Fitness: {model_results.results_dict['fitness']:.4f}", flush=True)
        print("\n############# TRAIN RESULTS ################", flush=True)
        print(f"##### Num Of Images: {num_of_train_images}", flush=True)
        print(f"##### Average Matches IoU: {avg_train_matches_iou:.4f}", flush=True)
        print(f"##### Average Matches&Unmatched IoU: {avg_train_matches_unmatched_iou:.4f}", flush=True)
        print(f"##### Average Unmatches Predictions: {avg_train_unmatches_pred_boxes:.4f}", flush=True)
        print(f"##### Average Unmatches Actual: {avg_train_unmatches_act_boxes:.4f}", flush=True)
        print("############################################", flush=True)           
    if config['test_model']:
        print("\n############# TEST RESULTS ################", flush=True)
        print(f"##### Num Of Images: {num_of_test_images}", flush=True)
        print(f"##### Average Matches IoU: {avg_test_matches_iou:.4f}", flush=True)
        print(f"##### Average Matches&Unmatched IoU: {avg_test_matches_unmatched_iou:.4f}", flush=True)
        print(f"##### Average Unmatches Predictions: {avg_test_unmatches_pred_boxes:.4f}", flush=True)
        print(f"##### Average Unmatches Actual: {avg_test_unmatches_act_boxes:.4f}", flush=True)
        print("############################################", flush=True)    

    end_time = datetime.now()
    print(f"\nTraining completed at: {end_time}", flush=True)
    print(f"Total training time: {end_time - start_time}", flush=True)        
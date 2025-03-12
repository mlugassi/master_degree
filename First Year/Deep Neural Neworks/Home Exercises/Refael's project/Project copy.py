from Utils import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
def evaluate_model(model, test_images_path, labelme_annotations_path, logger, iou_threshold=0.5):
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
            # print(f"Missing annotation for {img_path}, skipping.")
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
    print(f"\nFinal Model IOU Score - {test_images_path}: {overall_iou:.4f}")
    
    logger.report_scalar(title="IOU", series=f"{test_images_path}", value=overall_iou, iteration=epoch+1)

    return overall_iou

def predict_and_generate_json(model, image_folder, output_json_folder, confidence_threshold=0.5):
    """
    מקבל תיקיית תמונות, מריץ עליהן את המודל ומייצר קובצי JSON בפורמט של LabelMe.
    
    :param model: מודל Faster R-CNN מאומן
    :param image_folder: תיקיית התמונות
    :param output_json_folder: תיקייה לשמירת קובצי ה-JSON
    :param confidence_threshold: סף ביטחון לניבויים (ברירת מחדל 0.5)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    transform = T.Compose([T.ToTensor()])
    
    os.makedirs(output_json_folder, exist_ok=True)
    
    for img_filename in os.listdir(image_folder):
        if not img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(image_folder, img_filename)
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(image_tensor)
        
        predicted_boxes = prediction[0]['boxes'].cpu().numpy()
        predicted_scores = prediction[0]['scores'].cpu().numpy()
        
        labelme_data = {
            "version": "4.5.9",
            "flags": {},
            "shapes": [],
            "imagePath": img_filename,
            "imageData": None,
            "imageHeight": image.height,
            "imageWidth": image.width
        }
        
        for box, score in zip(predicted_boxes, predicted_scores):
            if score < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = box
            shape = {
                "label": "scroll_segment",
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            labelme_data["shapes"].append(shape)
        
        json_filename = os.path.splitext(img_filename)[0] + ".json"
        json_path = os.path.join(output_json_folder, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=4, ensure_ascii=False)
        
        print(f"Saved: {json_path}")

def create_fasterrcnn_model(pretrained):
    
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    return model

def create_yolo_model(pretrained):
    
    from ultralytics import YOLO
    if pretrained:
        model = YOLO("yolov8n.pt")  # ניתן להחליף לגרסאות אחרות כמו 'yolov8m.pt', 'yolov8x.pt' וכו'
    else: 
        model = YOLO(Models.yolo)

    return model

def create_retinanet_model(pretrained, num_classes=2):
    from torchvision.models.detection import retinanet_resnet50_fpn
    model = retinanet_resnet50_fpn(pretrained_backbone=pretrained,  num_classes=num_classes)

    return model

def create_efficientdet_model(pretrained):

    from effdet import get_efficientdet_config, EfficientDet

    config = get_efficientdet_config("tf_efficientdet_d0")
    model = EfficientDet(config)
    return model

class Models:
    faster_rcnn     = "./faster_rcnn.2.pth"
    yolo            = "./yolo.pth"
    retinanet       = "./retinanet.1.pth"
    efficientdet    = "./efficientdet"

# 1. batch:2, lr: 0.0001, epochs: 5,0000, pretrained: True
# 2. batch:1, lr: 0.0001, epochs: 5,0000, pretrained: True
# 3. batch:2, lr: 0.0001, epochs: 5,0000, pretrained: False
# 4. batch:2, lr: 0.001,  epochs: 5,0000, pretrained: True
# 5. batch:1, lr: 0.0001, epochs: 10,0000, pretrained: True

if __name__ == "__main__":
    # inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_dir = "train"  # Path to LabelMe JSON files
    img_dir = "train"  # Path to images
    num_classes = 2  # Background + scroll segment
    batch = 2
    model_name = Models.faster_rcnn
    lr = 0.0001
    num_epochs = 5*10

    dataset = ScrollDataset(json_dir, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # extract model
    if os.path.exists(model_name + ".gz"):
        extract_gz(model_name)
    elif glob.glob(f"{model_name}.part*.gz"):
        merge_and_decompress_pth(model_name)
    
    pretrained = False
    if not os.path.exists(model_name):
        pretrained = True
    
    params = {
        "batch": batch,
        "epochs": num_epochs,
        "learning_rate": lr,
        "model_name": model_name,
        "pretrained": pretrained
    }
    task = Task.init(project_name="DNN", task_name=f"Project")
    task.connect(params)
    logger = task.get_logger()

    print("###########################################")
    for key in params:
        print(f"# {key}: {params[key]}")
    print("###########################################")

    if model_name == Models.faster_rcnn:
        model = create_fasterrcnn_model(pretrained)
    elif model_name == Models.yolo:
        model = create_yolo_model(pretrained)
    elif model_name == Models.retinanet:
        model = create_retinanet_model(pretrained, num_classes)
    elif model_name == Models.efficientdet:
        model = create_efficientdet_model(pretrained)
    else:
        pass
    
    # Load pre-trained Faster R-CNN
    if not pretrained:
        model.load_state_dict(torch.load(model_name))

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

    # Save trained model
    torch.save(model.state_dict(), model_name)
    print("Model saved!")

    # הערכת המודל על סט הבדיקה
    with torch.no_grad():
        evaluate_model(model, "train",  "train",    logger)
        evaluate_model(model, "test",   "test",     logger)

    split_and_compress_pth(model_name)

    # predict_and_generate_json(model, "test", "test")



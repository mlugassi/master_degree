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
def evaluate_model(model, test_images_path, labelme_annotations_path, results = "./results", model_name = None, iou_threshold=0.5, last_epoch=False):
    model.eval()  # מצב הערכה (ללא גרדיאנטים)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if last_epoch:
        csv_dir_res = f"{results}/csv"
        img_dir_res = f"{results}/images"
        model_name = model_name.replace("./","")
        os.makedirs(csv_dir_res, exist_ok=True)
        os.makedirs(img_dir_res, exist_ok=True)

    transform = T.Compose([T.ToTensor()])
    total_iou = []
    
    # טעינת התמונות
    image_files = glob.glob(os.path.join(test_images_path, "*.jpg"))
    
    for img_path in image_files:
        if last_epoch:
            img_name = os.path.basename(img_path)
            csv_path = f"{csv_dir_res}/" + img_name.replace(".jpg",f".{model_name}.csv")
            img_output_path = f"{img_dir_res}/" + img_name.replace(".jpg",f".{model_name}.jpg")
            predict_process_bounding_boxes(img_path, csv_path)
            draw_bounding_boxes(img_path, csv_path, img_output_path)

        json_name = os.path.basename(img_path).replace(".jpg", ".json")
        annotation_path = os.path.join(labelme_annotations_path, json_name)

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

    return overall_iou

# faster_rcnn
# 1. batch:2, lr: 0.0001, epochs: 5,0000,  pretrained: True
# 2. batch:1, lr: 0.0001, epochs: 5,0000,  pretrained: True
# 3. batch:2, lr: 0.0001, epochs: 5,0000,  pretrained: False
# 4. batch:2, lr: 0.001,  epochs: 5,0000,  pretrained: True
# 5. batch:2, lr: 0.0001, epochs: 10,0000, pretrained: True

# retinanet
# 1. batch:2, lr: 0.0001, epochs: 5,0000,  pretrained: True
# 2. batch:1, lr: 0.0001, epochs: 5,0000,  pretrained: True
# 3. batch:2, lr: 0.0001, epochs: 5,0000,  pretrained: False
# 4. batch:2, lr: 0.001,  epochs: 5,0000,  pretrained: True
# 5. batch:2, lr: 0.0001, epochs: 10,0000, pretrained: True


if __name__ == "__main__":
    # inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_dir = "train"  # Path to LabelMe JSON files
    img_dir = "train"  # Path to images
    num_classes = 2  # Background + scroll segment
    # batch = 2
    # model_name = Models.retinanet
    # lr = 0.0001
    # num_epochs = 10*1000
    # image_path = "./test/M41139-1-C.jpg"
    # predict_process_bounding_boxes(image_path, "res.csv")
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_1, 2, 0.0001, 3*1000, True  # --> 15321
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_2, 1, 0.0001, 3*1000, True  # --> 15333
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_3, 2, 0.0001,  3*1000, False # --> 15326
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_4, 2, 0.001,  3*1000, True  # --> 15324
    # model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_5, 2, 0.0001, 6*1000, True  # --> 15325
    model_name, batch, lr, num_epochs, pretrained = Models.faster_rcnn_6, 1, 0.0001, 10*1000, True  # --> 15336
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_2,   2, 0.0001, 3*1000, True  # --> 15327
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_1,   1, 0.0001, 3*1000, True  # --> 15328
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_3,   2, 0.0001, 3*1000, False # --> 15329
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_4,   2, 0.001,  3*1000, True  # --> 15330
    # model_name, batch, lr, num_epochs, pretrained = Models.retinanet_5,   2, 0.0001, 6*1000, True  # --> 15331


    dataset = ScrollDataset(json_dir, img_dir)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # extract model
    if os.path.exists(model_name + ".gz"):
        extract_gz(model_name)
    elif glob.glob(f"{model_name}.part*.gz"):
        merge_and_decompress_pth(model_name)
    
    if pretrained not in [True, False]:
        if os.path.exists(model_name):
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
    if not pretrained and os.path.isfile(model_name):
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
        logger.report_scalar(title="Loss", series="Train", value=loss.item(), iteration=epoch+1)

        # Save trained model
        if epoch % 100 == 0:
            torch.save(model.state_dict(), model_name)
            print(f"Model {model_name} saved!")

            with torch.no_grad():
                iou_train = evaluate_model(model=model, test_images_path="train",  labelme_annotations_path="train", model_name=model_name)
                iou_test  = evaluate_model(model=model, test_images_path="test",   labelme_annotations_path="test",  model_name=model_name)
                logger.report_scalar(title="IOU", series="Train", value=iou_train, iteration=epoch)
                logger.report_scalar(title="IOU", series="Test",  value=iou_test,  iteration=epoch)
            model.train()

    torch.save(model.state_dict(), model_name)
    print(f"Model {model_name} saved!")
    with torch.no_grad():
        iou_train = evaluate_model(model=model, test_images_path="train",  labelme_annotations_path="train", model_name=model_name, last_epoch=True)
        iou_test  = evaluate_model(model=model, test_images_path="test",   labelme_annotations_path="test",  model_name=model_name, last_epoch=True)
        logger.report_scalar(title="IOU", series="Train", value=iou_train, iteration=epoch)
        logger.report_scalar(title="IOU", series="Test",  value=iou_test,  iteration=epoch)

    split_and_compress_pth(model_name)

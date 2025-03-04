import torch
import numpy as np
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 💡 שלב 1: טוען את מודל YOLOv8 לזיהוי מגילות
yolo_model = YOLO("yolov8m.pt")  # מודל בינוני (אפשר גם yolov8s.pt לגרסה קטנה יותר)

# import requests

# url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h.pth"
# filename = "sam_vit_h.pth"

# response = requests.get(url, stream=True)
# with open(filename, "wb") as file:
#     for chunk in response.iter_content(chunk_size=1024):
#         if chunk:
#             file.write(chunk)

# print("✅ SAM Model Downloaded Successfully!")



# 💡 שלב 2: טוען את מודל SAM (Segment Anything Model)
sam_checkpoint = "sam_vit_h.pth"  # יש להוריד את המודל מהאינטרנט
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 💡 שלב 3: טוען תמונה ומבצע זיהוי באמצעות YOLO
image_path = "train\M42441-1-E.jpg"
image = cv2.imread(image_path)

# מריץ את YOLO
yolo_results = yolo_model(image)

# לוקח את התיבות החוסמות של YOLO
bounding_boxes = []
for result in yolo_results:
    for box in result.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)  # המרה למספרים שלמים
        bounding_boxes.append((x1, y1, x2, y2))

# 💡 שלב 4: משתמש ב-SAM כדי לשפר את הזיהוי
predictor.set_image(image)

# יוצר מסכות סביב כל תיבה שזוהתה ע"י YOLO
final_bounding_boxes = []
for (x1, y1, x2, y2) in bounding_boxes:
    box = np.array([[x1, y1, x2, y2]])
    masks, _, _ = predictor.predict(box=box)

    # מוצא Bounding Box מתוך המסכה של SAM
    for mask in masks:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        final_bounding_boxes.append((x, y, x + w, y + h))

# 💡 שלב 5: מצייר את התיבות החוסמות
for (x1, y1, x2, y2) in final_bounding_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # תיבה אדומה

cv2.imwrite(image_path.replace(".jpg", "_out.jpg"), image)  # שומר את התמונה עם התיוגים
print("✅ תיוג הושלם! התמונה נשמרה בשם output.jpg")

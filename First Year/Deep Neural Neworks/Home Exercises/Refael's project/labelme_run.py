import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

# טען את מודל YOLOv8 המוכן (נשתמש ב-yolov8n.pt שהוא הקל ביותר)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = YOLO("yolov8n.pt")

# נתיב לתיקיית התמונות
input_folder = "train"
output_folder = "train_labeled"

# צור תיקיית פלט אם היא לא קיימת
os.makedirs(output_folder, exist_ok=True)

# עבור על כל התמונות בתיקייה
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):  # בדיקה שהתמונה בפורמט מתאים
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        # זיהוי אובייקטים בתמונה
        results = model(img)[0]

        # יצירת רשימת תיבות תוחמות (bounding boxes)
        shapes = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()  # קואורדינטות ותוצאה מהמודל
            label = results.names[int(cls)]  # שם האובייקט שזוהה

            # יצירת תיוג בפורמט LabelMe
            shape = {
                "label": label,
                "points": [[x1, y1], [x2, y2]],  # תיבת התיחום
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)

        # יצירת קובץ JSON בפורמט של LabelMe
        json_data = {
            "version": "5.1.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": filename,
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1]
        }

        # שמירת JSON עם אותו שם של התמונה
        json_filename = os.path.join(output_folder, filename.replace(".", "_") + ".json")
        with open(json_filename, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"✅ תויגה תמונה: {filename}")

print("🎯 כל התמונות תויגו ונשמרו בתיקייה:", output_folder)

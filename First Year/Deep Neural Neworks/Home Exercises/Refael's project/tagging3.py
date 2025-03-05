import os
import subprocess
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 📌 שלב 1: התקנת התלויות אם חסרות
try:
    import groundingdino
except ImportError:
    print("🔴 Grounding DINO not found! Installing now...")
    subprocess.run(["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"])
    subprocess.run(["pip", "install", "-e", "GroundingDINO"])
    print("✅ Grounding DINO installed!")

# 📌 שלב 2: התקנת תלויות נוספות
subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio", "transformers", "numpy", "opencv-python", "matplotlib", "pycocotools"])

# 📌 שלב 3: הורדת המודלים אם חסרים
def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"📥 Downloading {dest}...")
        subprocess.run(["wget", "-O", dest, url])
    else:
        print(f"✅ {dest} already exists, skipping download.")

# הורדת Grounding DINO אם חסר
dino_checkpoint = "GroundingDINO/groundingdino_swint_ogc.pth"
download_file("https://drive.google.com/uc?id=1R37jZ7oe-AhDrL7TnD6oKA5PLrDMr4kO", dino_checkpoint)

# הורדת SAM אם חסר
sam_checkpoint = "sam_vit_h.pth"
download_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h.pth", sam_checkpoint)

# 📌 שלב 4: ייבוא המודלים
from groundingdino.util.inference import Model as GroundingDINO
from segment_anything import sam_model_registry, SamPredictor

# טוען את המודלים
dino_model = GroundingDINO(model_config_path="GroundingDINO/config.py", model_checkpoint=dino_checkpoint)
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
sam_predictor = SamPredictor(sam)

# 📌 שלב 5: טוען תמונה
image_path = "your_image.jpg"  # שנה את הנתיב לתמונה שלך
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 📌 שלב 6: הפעלת Grounding DINO לזיהוי מגילות בתמונה
prompt = "scroll fragment, ancient manuscript, parchment"
boxes, logits, phrases = dino_model.predict_with_caption(image_rgb, caption=prompt, box_threshold=0.3, text_threshold=0.25)

# 📌 שלב 7: שיפור הזיהוי עם SAM
sam_predictor.set_image(image_rgb)
final_boxes = []
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    masks, _, _ = sam_predictor.predict(boxes=np.array([[x1, y1, x2, y2]]))

    for mask in masks:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        final_boxes.append((x, y, x + w, y + h))

# 📌 שלב 8: ציור התוצאות ושמירתן
for (x1, y1, x2, y2) in final_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # מסגרת אדומה

cv2.imwrite("output.jpg", image)
print("✅ תיוג הושלם! התמונה נשמרה בשם output.jpg")

# הצגת התמונה עם Bounding Boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

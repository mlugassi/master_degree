import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 💡 טוען את מודל CLIPSeg
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# 💡 טוען תמונה
image_path = "train\M42441-1-E.jpg"
image = Image.open(image_path).convert("RGB")

# 💡 טקסט שמגדיר מה לזהות
# prompt = ["scroll fragment"]  # אפשר להוסיף מילים אחרות כדי לשפר דיוק
prompt = ["scroll fragment", "ancient manuscript", "parchment", "scroll", "handwritten text"]

# מעבד את התמונה והטקסט
inputs = processor(text=prompt, images=image, return_tensors="pt")

# מקבל מסיכה של האובייקט המתאים לטקסט
with torch.no_grad():
    outputs = model(**inputs)
    mask = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

# 💡 מסנן ומגדיר סף למסיכה (Thresholding)
mask = (mask > 0.5).astype(np.uint8)  # מסנן אזורים עם הסתברות גבוהה בלבד

# 💡 ממיר את המסיכה לתיבות חוסמות (Bounding Boxes) עם OpenCV
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_cv = np.array(image)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)  # תיבה כחולה

# 💡 שמירת התוצאה
cv2.imwrite(image_path.replace(".jpg", "_output2.jpg"), image_cv)
print("✅ תיוג הושלם! התמונה נשמרה בשם ", os.path.realpath(image_path.replace(".jpg", "_output2.jpg")))

# הצגת התמונה עם המסיכות
plt.imshow(image_cv)
plt.axis("off")
plt.show()
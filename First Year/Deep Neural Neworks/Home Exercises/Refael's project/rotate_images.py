import os
import json
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = "test2"
OUTPUT_DIR = f"{INPUT_DIR}_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def rotate_image(image, angle):
    """ מסובב תמונה בזווית נתונה """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def rotate_bounding_boxes(boxes, angle, img_w, img_h):
    """ מסובב תיבות חוסמות בהתאם לזווית ולגודל התמונה """
    rotated_boxes = []
    for xmin, ymin, xmax, ymax in boxes:
        if angle == 270:
            new_xmin = ymin
            new_ymin = img_w - xmax
            new_xmax = ymax
            new_ymax = img_w - xmin
        elif angle == 180:
            new_xmin = img_w - xmax
            new_ymin = img_h - ymax
            new_xmax = img_w - xmin
            new_ymax = img_h - ymin
        elif angle == 90:
            new_xmin = img_h - ymax
            new_ymin = xmin
            new_xmax = img_h - ymin
            new_ymax = xmax
        rotated_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
    return rotated_boxes

def draw_bounding_boxes(image, boxes, output_path):
    """ מצייר Bounding Boxes על התמונה ושומר אותה בשם חדש """
    for (xmin, ymin, xmax, ymax) in boxes:
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)  # ירוק

    bbox_output_path = output_path.replace(".jpg", "_bbox.jpg")
    cv2.imwrite(bbox_output_path, image)
    print(f"✔ נשמר עם תיבות חוסמות: {bbox_output_path}")

# עיבוד כל התמונות בתיקייה
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".jpg"):
        img_path = os.path.join(INPUT_DIR, filename)
        json_path = img_path.replace(".jpg", ".json")

        if not os.path.exists(json_path):
            print(f"❌ אין JSON תואם עבור {filename}, מדלג...")
            continue

        # טוען תמונה
        image = cv2.imread(img_path)
        img_h, img_w = image.shape[:2]

        # טוען קובץ JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # שליפת התיבות החוסמות מה-JSON
        boxes = [shape["points"][0] + shape["points"][1] for shape in data["shapes"]]

        for angle in [90, 180, 270]:
            rotated_img = rotate_image(image, angle)
            rotated_boxes = rotate_bounding_boxes(boxes, angle, img_w, img_h)

            # שם קבצים חדשים
            base_name = filename.replace(".jpg", f"_rotated_{angle}")
            rotated_img_path = os.path.join(OUTPUT_DIR, base_name + ".jpg")
            rotated_json_path = os.path.join(OUTPUT_DIR, base_name + ".json")

            # שמירת התמונה המסובבת
            cv2.imwrite(rotated_img_path, rotated_img)

            # עדכון ה-JSON ושמירתו
            new_json_data = data.copy()
            for i, shape in enumerate(new_json_data["shapes"]):
                shape["points"] = [[rotated_boxes[i][0], rotated_boxes[i][1]],
                                   [rotated_boxes[i][2], rotated_boxes[i][3]]]

            with open(rotated_json_path, "w", encoding="utf-8") as f:
                json.dump(new_json_data, f, indent=4, ensure_ascii=False)

            # יצירת תמונה עם Bounding Boxes
            # draw_bounding_boxes(rotated_img.copy(), rotated_boxes, rotated_img_path)

print("🎯 הסתיים! כל התמונות והקבצים המסובבים נשמרו בהצלחה.")

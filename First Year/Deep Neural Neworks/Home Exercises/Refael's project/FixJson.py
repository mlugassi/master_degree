import json
import math
import os
import glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_labelme_json(json_path):
    """ טוען קובץ JSON של LabelMe """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_labelme_json(data, output_path):
    """ שומר את קובץ ה-JSON לאחר עיבוד """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def calculate_distance_to_top_left(bbox):
    """ מחזיר את המרחק של הפינה השמאלית העליונה של התיבה מנקודת (0,0) """
    x_min, y_min, x_max, y_max = bbox
    return math.sqrt(x_min ** 2 + y_min ** 2)  # מרחק אוקלידי מהפינה השמאלית העליונה

def sort_labelme_bounding_boxes(json_path, output_path):
    """ ממיין את התיבות החוסמות בקובץ JSON ושומר את הקובץ מחדש """
    data = load_labelme_json(json_path)
    
    for i, shape in enumerate(data["shapes"]):
        new_shape = shape
        x_min = min(shape["points"][0][0], shape["points"][1][0])
        y_min = min(shape["points"][0][1], shape["points"][1][1])
        x_max = max(shape["points"][0][0], shape["points"][1][0])
        y_max = max(shape["points"][0][1], shape["points"][1][1])
        shape["points"] = [[x_min,y_min], [x_max,y_max]]

    # סידור הרשימה לפי המרחק לפינה השמאלית העליונה
    data["shapes"].sort(key=lambda shape: calculate_distance_to_top_left([
        min(p[0] for p in shape["points"]),  # x_min
        min(p[1] for p in shape["points"]),  # y_min
        max(p[0] for p in shape["points"]),  # x_max
        max(p[1] for p in shape["points"])   # y_max
    ]))

    # שמירת הקובץ לאחר מיון
    save_labelme_json(data, output_path)
    print(f"✅ קובץ JSON מוין ונשמר בהצלחה ל-{output_path}")

# 📌 דוגמה לשימוש
json_files = glob.glob(os.path.join("./train", "*.json"))
for json_file in json_files:
    sort_labelme_bounding_boxes(json_file, json_file)
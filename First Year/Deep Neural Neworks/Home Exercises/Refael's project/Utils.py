import torch
import torchvision
import json
import os
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import numpy as np
import torchvision.transforms as T
from PIL import Image
import glob
import torch
import torchvision
import gzip
import shutil
import re

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    return obj

def extract_gz(gz_file, output_file=None):
    if not gz_file.endswith(".gz"):
        raise ValueError("Not a .gz file")

    if output_file is None:
        output_file = gz_file[:-3]  # מסיר את הסיומת .gz
    with gzip.open(gz_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Extracted: {gz_file} -> {output_file}")
    return output_file

def compress_to_gz(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + ".gz"  # מוסיף סיומת .gz אם לא סופק שם פלט

    with open(input_file, 'rb') as f_in, gzip.open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"Compressed: {input_file} -> {output_file}")
    return output_file

def split_and_compress_pth(input_file, max_size=50 * 1024 * 1024):  # גודל מרבי לכל חלק (כאן 100MB)
    part_num = 0
    with open(input_file, 'rb') as f_in:
        while True:
            chunk = f_in.read(max_size)
            if not chunk:
                break
            part_num += 1
            part_filename = f"{input_file}.part{part_num}.gz"
            with gzip.open(part_filename, 'wb') as f_out:
                f_out.write(chunk)
            print(f"Created: {part_filename}")

def merge_and_decompress_pth(input_file):
    # חיפוש כל הקבצים עם התבנית המתאימה
    part_files = sorted(glob.glob(f"{input_file}.part*.gz"), key=lambda x: int(re.search(r'\d+', x).group()))
    
    if not part_files:
        print("No parts found!")
        return
    
    with open(input_file, 'wb') as f_out:
        for part_filename in part_files:
            with gzip.open(part_filename, 'rb') as f_in:
                f_out.write(f_in.read())
            print(f"Merged: {part_filename}")

    print(f"Reconstructed file saved as {input_file}")

# פונקציה לחישוב IOU בין שתי תיבות חוסמות
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # חישוב חיתוך
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # חישוב איחוד
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0
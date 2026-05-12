import os
import shutil
import csv
import cv2
import numpy as np

def get_bounding_box(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0.5, 0.5, 0.9, 0.9
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        img_h, img_w = img.shape[:2]
        center_x = (x + w / 2) / img_w
        center_y = (y + h / 2) / img_h
        width = w / img_w
        height = h / img_h
        
        if width > 0.05 and height > 0.05:
            return center_x, center_y, width, height
            
    return 0.5, 0.5, 0.9, 0.9

def process_split(split_name):
    base_dir = os.path.join("dataset", split_name)
    if not os.path.exists(base_dir):
        return
    
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    csv_file = os.path.join(base_dir, "_classes.csv")
    if not os.path.exists(csv_file):
        print(f"No _classes.csv found in {base_dir}")
        return
        
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        class_names = [c.strip() for c in header[1:]]
        
        for row in reader:
            filename = row[0].strip()
            labels = [int(v.strip()) for v in row[1:]]
            
            src_img = os.path.join(base_dir, filename)
            dst_img = os.path.join(images_dir, filename)
            
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
                
            img_path_for_bbox = dst_img if os.path.exists(dst_img) else src_img
            cx, cy, bw, bh = get_bounding_box(img_path_for_bbox)
            
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            dst_txt = os.path.join(labels_dir, txt_filename)
            
            with open(dst_txt, 'w') as out_f:
                for idx, is_present in enumerate(labels):
                    if is_present == 1:
                        out_f.write(f"{idx} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}\n")

for split in ["train", "valid", "test"]:
    process_split(split)

print("Dataset preparation complete.")

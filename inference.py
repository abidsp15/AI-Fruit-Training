from ultralytics import YOLO
import cv2
import os
import sys
import glob

def main():
    run_dirs = sorted(glob.glob('runs/detect/fruit_detector*'), key=os.path.getmtime)
    if not run_dirs:
        print("Error: No training runs found.")
        sys.exit(1)
    
    weights_path = os.path.join(run_dirs[-1], 'weights', 'best.pt')
    
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at {weights_path}.")
        sys.exit(1)

    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)

    image_dir = 'dataset/test/images'
    
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist.")
        sys.exit(1)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {image_dir}.")
        sys.exit(1)

    print(f"Found {len(image_files)} images in {image_dir}.")

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        
        results = model(img_path)
        annotated_image = results[0].plot()
        
        window_name = "Fruit Detection Preview"
        cv2.imshow(window_name, annotated_image)
        
        print(f"Displaying {img_name}... Press any key to continue, or 'q' to quit.")
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

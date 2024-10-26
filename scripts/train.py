import os
import cv2
import numpy as np

IMG_SIZE = (224, 224)
data_dir = './data/Problem_2/'


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image normalization
    gray = gray.astype('float32') / 255.0
    # edge detection (Canny)
    edges = cv2.Canny((gray * 255).astype(np.uint8), threshold1=50, threshold2=150)
    return edges

# Example of preprocessing images in the training set folder
for label in ['train', 'val', 'test']:
    for crack_label in ['Negative', 'Positive']:
        folder_path = os.path.join(data_dir, label, crack_label)
        for img_file in os.listdir(folder_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Check file format
                img_path = os.path.join(folder_path, img_file)
                processed_img = preprocess_image(img_path)
                
                # Optionally save the processed image to a new folder or overwrite
                cv2.imwrite(os.path.join(folder_path, f'processed_{img_file}'), processed_img)
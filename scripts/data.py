import os
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

image_dir = './dataset/MVI_Training_Datasets/Problem_2/'
data_dir = './data/Problem_2/'

# create output directories (train, test, and validation)
for split in ['train', 'test', 'val']:
    for label in ['Negative', 'Positive']:
        os.makedirs(os.path.join(data_dir, split, label), exist_ok=True)

labels = []
image_paths = []

for label in ['Negative', 'Positive']:
    label_dir = os.path.join(image_dir, label)
    for img_file in os.listdir(label_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(label_dir, img_file))
            labels.append(0 if label == 'Negative' else 1)

# Convert to numpy arrays
labels = np.array(labels)

# data splitting into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")

# Copy images to respective directories
for img_path, label in zip(X_train, y_train):
    shutil.copy(img_path, os.path.join(data_dir, 'train', 'Negative' if label == 0 else 'Positive'))

for img_path, label in zip(X_val, y_val):
    shutil.copy(img_path, os.path.join(data_dir, 'val', 'Negative' if label == 0 else 'Positive'))

for img_path, label in zip(X_test, y_test):
    shutil.copy(img_path, os.path.join(data_dir, 'test', 'Negative' if label == 0 else 'Positive'))
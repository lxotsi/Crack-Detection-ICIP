import pathlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Conv2D, Input, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from data import X_train, X_val, y_train, y_val, X_test, y_test

# Data visualization
crack_sample = cv2.imread('dataset/MVI_Training_Datasets/Problem_2/Positive/00010.jpg')
nocrack_sample = cv2.imread('dataset/MVI_Training_Datasets/Problem_2/Negative/00007.jpg')
print(crack_sample.shape)
print(nocrack_sample.shape)

# run using GPU
# Set the GPU memory growth option
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()

print("Number of replicas:", strategy.num_replicas_in_sync)
data_directory = pathlib.Path('dataset/MVI_Training_Datasets/Problem_2/')
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 25 * strategy.num_replicas_in_sync
# Positive: Crack
# Negative: No Crack
class_names = ['Positive', 'Negative']
IMG_SIZE = (227, 227)

def preprocess_image(image):
    # resize the image
    img = cv2.imread(image)
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram equalization
    gray = cv2.equalizeHist(gray)
    # Normalize intensity
    gray = gray.astype('float32') / 255.0  # Normalize to [0, 1]
    # Apply edge detection (using Canny as an example)
    edges = cv2.Canny((gray * 255).astype(np.uint8), threshold1=50, threshold2=150)
    # Optional: Apply Gaussian blur to smooth the edges
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return image

def eval(y_true, y_pred):
    # Convert probabilities to class predictions (assuming y_pred is probabilities)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate precision, recall, F1-score, and accuracy
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    accuracy = accuracy_score(y_true, y_pred_classes)

    return precision, recall, f1, accuracy

def compute_iou(y_true, y_pred):
    # Convert to binary masks (1 for Positive, 0 for Negative)
    y_true_binary = (y_true == 1).astype(np.uint8)
    y_pred_binary = (y_pred == 1).astype(np.uint8)

    intersection = np.sum(y_true_binary & y_pred_binary)
    union = np.sum(y_true_binary) + np.sum(y_pred_binary) - intersection

    iou = intersection / union if union != 0 else 0
    return iou

def resnet_model():
    input_tensor = Input(shape=(227, 227, 1))
    num_classes = 2
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_tensor)

    # adding batch normalization layers
    x = base_model.output
    x = BatchNormalization()(x)
    # reducing the dimensions by using Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    # fully connected layer with BN
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)

    # output layer (Binary classification)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

model = resnet_model()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Evaluating the model on the test set
y_true = np.argmax(y_test, axis=1)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
precision, recall, f1, accuracy = eval(y_true, y_pred)
iou = compute_iou(y_true, y_pred_classes)

report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'IoU: {iou:.4f}')
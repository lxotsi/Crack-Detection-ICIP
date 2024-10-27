# Aquatic - Problem 4
Problem-4: Detecting Aquatic Defects Using Underwater Imagery.

The health of aquatic environments is critical for ecological balance and human well-being. This conference challenge aims to address this concern by focusing on recognizing aquatic defects using underwater imagery. Participants will be provided with a specialized dataset comprising underwater images captured in various aquatic environments.


This repository contains code for training a deep learning model that uses image data to detect cracks in aquatic environments. The model leverages the ResNet50 architecture and implements custom loss functions to enhance performance.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lxotsi/Aquatic-Detection-ICIP.git
   cd Aquatic-Detection-ICIP

2. Install required packages:
   ```bash
   pip install -r requirements.txt

## Dataset Preparation
The dataset consists of images labeled by the presence or absence of cracks. The code processes the dataset by:

Loading images and their corresponding labels.
Splitting the dataset into training, validation, and test sets (80% / 10% / 10%), respectively.
Calculating class weights to handle class imbalance.

## Model Architecture
The model is built using the ResNet50 architecture pre-trained on ImageNet. The key components include:

Base Model: ResNet50 with weights initialized from ImageNet, excluding the top layer.
Custom Layers: Added layers for pooling, dropout, and dense output.
Focal Loss: A custom focal loss function to address class imbalance during training.

## Training and Evaluation
The model is trained using the following steps:

Load datasets into TensorFlow data pipelines.
Compile the model using the Adam optimizer and the custom focal loss.
Train the model with early stopping and model checkpointing.
After training, the model is evaluated on the test set. Performance metrics calculated include:

- Accuracy
- Precision
- Recall
- F1-Score
- Additionally, a confusion matrix is generated to visualize performance.

## Results
Upon evaluation, the following metrics are reported:

- Accuracy: 
- Precision: 
- Recall: 
- F1-Score: 
- Confusion matrix visualizations are displayed using Matplotlib.

## Dependencies
This project requires the following Python packages:

- tensorflow
- numpy
- pandas
- opencv-python
- matplotlib
- seaborn
- scikit-learn

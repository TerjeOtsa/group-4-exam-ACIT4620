Emotion Detection using Convolutional Neural Networks
Overview
This project involves a series of Python scripts using PyTorch, OpenCV, and other libraries to create and deploy a deep learning model for emotion detection from images. The scripts cover data preprocessing, model training and evaluation, real-time emotion detection using webcam, and data analysis.

Contents
Data Preprocessing and Model Training (script1.py): Handles loading, transforming, and splitting image data into training, validation, and test sets. It defines and trains a CNN for emotion classification.
Enhanced Model Definition (script2.py): An improved version of the emotion classification model with additional layers and batch normalization.
Real-Time Emotion Detection (script3.py): Utilizes a webcam to capture images in real-time, preprocesses them, and uses the trained model to predict and display emotions on the screen.
two models are currently added to the git hub (emotion_classifier_0.5drop.pth) and (emotion_classifier100e.pth)
The lastest training history (training_history.json) and the first 100 epoch training in (training_history100.json) contains all the metrics collected from training the model.
figure_1.png is the metrics over 100 epochs
Requirements
Python 3.x
PyTorch
OpenCV
Matplotlib
NumPy
PIL
scikit-learn
dataset downloaded from https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset
INSTALLATION 
Before running the scripts, ensure that all required dependencies are installed. You can install them using pip:
pip install torch torchvision opencv-python numpy matplotlib Pillow scikit-learn

USAGE
Data Preprocessing and Model Training
Prepare your dataset in the required format as described in nn3.py.
Run the script using:
python nn3.py
This script will train the model and plot the training and validation loss.

Model to import into livefeed.py
This script (NeuralNetwork.py) is only used to import the NeuralNetwork into the live feed script
Real-Time Emotion Detection
Ensure the trained model file (e.g., emotion_classifier100e.pth) is in the same directory.

Run the real-time detection script:
python livefeed.py
This will open your webcam and start detecting emotions in real-time.

Model Details
The CNN model designed for emotion classification includes convolutional layers, batch normalization, pooling layers, fully connected layers, and dropout for regularization.

Dataset Structure
The dataset should consist of images classified into different emotions. The expected structure is a directory for each emotion containing its respective images.

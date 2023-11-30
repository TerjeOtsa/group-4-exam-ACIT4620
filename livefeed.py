import cv2
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
from NeuralNetwork import EmotionClassifier

# --- Section 1: Model Loading ---
# Load the trained model
model = EmotionClassifier()  
model.load_state_dict(torch.load('emotion_classifier100edrop.pth'))
model.eval()

# --- Section 2: Setup ---
# Define the class labels (emotions) as used during training
emotion_labels = ['happy', 'sad', 'angry', 'surprise', 'neutral', 'disgust', 'fear'] 

# Define transformations as used during training for the input images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),  # Assuming 48x48 as in your training script
    transforms.ToTensor(),
    transforms.Lambda(lambda x: transforms.functional.normalize(x, [x.mean()], [x.std()]))        
])

# --- Section 3: Image Preprocessing Function ---
# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = transform(image).unsqueeze(0)
    return image

# --- Section 4: Video Capture and Emotion Detection ---
# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    processed_frame = preprocess_image(frame)

    # Make prediction
    with torch.no_grad():
        outputs = model(processed_frame)
        _, predicted = torch.max(outputs, 1)
        emotion = emotion_labels[predicted.item()]

    # Display the resulting frame with predicted emotion
    cv2.putText(frame, f'Emotion: {emotion}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Section 5: Cleanup ---
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
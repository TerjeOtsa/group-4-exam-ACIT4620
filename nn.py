import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns  

# Define paths and parameters
train_path = 'images/train'
validation_path = 'images/validation'
img_size = 48
batch_size = 64
epochs = 20
num_classes = 7  # Assuming 7 different emotions

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load datasets with initial transform
print("Loading datasets...")
train_val_dataset = datasets.ImageFolder(root=train_path, transform=transform)
validation_dataset = datasets.ImageFolder(root=validation_path, transform=transform)
print("Datasets loaded successfully.")

# Define the proportion of the test set
test_proportion = 0.2

# Calculate the number of samples for test and train sets
num_train_samples = int((1.0 - test_proportion) * len(train_val_dataset))
num_test_samples = len(train_val_dataset) - num_train_samples

# Split the dataset
train_subset, test_subset = random_split(train_val_dataset, [num_train_samples, num_test_samples])

# Define loaders using the subsets
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Compute mean and std for normalization using only the train subset
def compute_mean_std(loader):
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

print("Computing mean and standard deviation for normalization...")
train_loader_for_mean_std = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
mean, std = compute_mean_std(train_loader_for_mean_std)
print(f"Computed mean: {mean}, std: {std}")

# Update the transform with normalization using computed mean and std
print("Updating transformations with normalization...")
transform_normalized = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Apply updated transformations to the datasets
print("Applying updated transformations to datasets...")
train_val_dataset.transform = transform_normalized
validation_dataset.transform = transform_normalized

# Redefine the DataLoaders with the updated datasets
print("Loading and preparing data loaders...")
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Define the CNN model
print("Initializing the model...")
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # Pooling, activation, and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Instantiate the model
model = EmotionClassifier()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Function to train the model
def train_model(model, criterion, optimizer, train_loader, validation_loader, epochs):
    print("Starting training...")
    model.train()
    training_loss = []
    validation_loss = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        training_loss.append(running_loss / len(train_loader))
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validation_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        validation_loss.append(val_loss / len(validation_loader))
        
        print(f'Epoch {epoch+1}/{epochs} training loss: {training_loss[-1]} validation loss: {validation_loss[-1]}')
        
    return training_loss, validation_loss

# Train the model
print("Training in progress...")
training_loss, validation_loss = train_model(model, criterion, optimizer, train_loader, validation_loader, epochs)
print("Training completed.")
# Plot the training and validation loss
print("Plotting the loss curves...")
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

def evaluate_model(model, data_loader, dataset_name="Validation"):
    print(f"Evaluating on {dataset_name} set...")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=data_loader.dataset.classes)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Display the metrics
    print(f'Accuracy: {accuracy}\n')
    print('Classification Report:')
    print(class_report)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Optional: Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=data_loader.dataset.classes,
                yticklabels=data_loader.dataset.classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {dataset_name} Set')
    plt.show()

# Evaluate on the validation set
print("Evaluating on validation set:")
evaluate_model(model, validation_loader)

# Evaluate on the test set
print("Evaluating on test set:")
evaluate_model(model, test_loader)

# Plot the training and validation loss
print("Plotting the loss curves...")
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
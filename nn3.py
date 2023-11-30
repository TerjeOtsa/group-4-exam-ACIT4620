# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import time
import json


# --- Section 1: Imports and Initial Setup ---
# Define paths and parameters
train_path = 'images/train'
validation_path = 'images/validation'
img_size = 48
batch_size = 64
epochs = 100
num_classes = 7  # Assuming 7 different emotions


# --- Section 2: Dataset Preparation and Preprocessing ---
# Load datasets
dataset = datasets.ImageFolder(root=train_path)

# Calculate class weights
class_counts = {label: 0 for label, _ in dataset.class_to_idx.items()}
for _, index in dataset.samples:
    label = dataset.classes[index]
    class_counts[label] += 1
class_weights = [1.0 / class_counts[class_] for class_ in dataset.classes]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Define transformations with data augmentation for training data
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
    # Normalization will be added after computing the mean and std
])

# Transforms without augmentation for validation and test data
validation_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
    # Normalization will be added after computing the mean and std
])

# Load datasets with initial transform (without normalization)
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
validation_dataset = datasets.ImageFolder(root=validation_path, transform=validation_transforms)

# Split the training dataset
num_train_samples = int((1.0 - 0.2) * len(train_dataset))
num_test_samples = len(train_dataset) - num_train_samples
train_subset, test_subset = random_split(train_dataset, [num_train_samples, num_test_samples])
print(f"Training set size: {len(train_subset)}")
print(f"Validation set size: {len(validation_dataset)}")
print(f"Test set size: {len(test_subset)}")

# Compute mean and std for normalization
def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std

train_loader_for_mean_std = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
mean, std = compute_mean_std(train_loader_for_mean_std)

# --- Section 3: Data Visualization ---
# Define the function right after the datasets are loaded
def plot_images(dataset, num_images=5, cols=7):
    fig, axes = plt.subplots(nrows=num_images, ncols=cols, figsize=(15, num_images * 3))
    classes = dataset.classes
    for i in range(num_images):
        for j in range(cols):
            idx = np.random.choice(np.where(np.array(dataset.targets) == j)[0])
            img, label = dataset[idx]
            ax = axes[i][j]
            ax.imshow(img.permute(1, 2, 0).squeeze(), cmap="gray")
            ax.axis('off')
            if i == 0:
                ax.set_title(classes[label])
    plt.tight_layout()
    plt.show()

plot_images(train_dataset)


# Update transforms with normalization
train_transforms.transforms.append(transforms.Normalize(mean, std))
validation_transforms.transforms.append(transforms.Normalize(mean, std))
train_dataset.transform = train_transforms
validation_dataset.transform = validation_transforms

# DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


# --- Section 4: Model Definition ---
# Define the CNN model
print("Initializing the model...")
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 1024) #image size 48 (256 * 8 * 8, 1024) for 128
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Pooling, activation, and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    

# --- Section 5: Hyperparameter Setup ---
# Define hyperparameters
learning_rate = 0.0015  # You can change this as needed
betas = (0.9, 0.999)   # Default values for Adam optimizer
weight_decay = 0.001     # You can change this as needed

# Instantiate the model, define loss and optimizer
model = EmotionClassifier()
print(model)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

#define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
# Training function


# --- Section 6: Training Function Implementation ---
# Define the train function 
def train_model(model, criterion, optimizer, train_loader, validation_loader, epochs):
    best_val_f1 = 0.0  # Initialize the best validation F1 score
    model_save_path = 'emotion_classifier.pth'  # Define your path here
    history = {'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
   
    for epoch in range(epochs):
        # Variables to track performance metrics
        train_loss = 0.0
        valid_loss = 0.0
        train_corrects = 0
        valid_corrects = 0
        train_preds = []
        train_labels = []
        valid_preds = []
        valid_labels = []

        # Training phase
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='macro', zero_division=0)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_corrects += torch.sum(preds == labels.data)
                valid_preds.extend(preds.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())

        valid_loss /= len(validation_loader.dataset)
        valid_acc = valid_corrects.double() / len(validation_loader.dataset)
        valid_precision, valid_recall, valid_f1, _ = precision_recall_fscore_support(valid_labels, valid_preds, average='macro', zero_division=0)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['train_precision'].append(train_precision)
        history['train_recall'].append(train_recall)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc.item())
        history['val_precision'].append(valid_precision)
        history['val_recall'].append(valid_recall)
        history['val_f1'].append(valid_f1)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Training Loss: {train_loss:.4f} Acc: {train_acc:.4f} Precision: {train_precision:.4f} Recall: {train_recall:.4f} F1: {train_f1:.4f}')
        print(f'Validation Loss: {valid_loss:.4f} Acc: {valid_acc:.4f} Precision: {valid_precision:.4f} Recall: {valid_recall:.4f} F1: {valid_f1:.4f}\n')
        current_val_f1 = history['val_f1'][-1]
        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            # Save the model at this state
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at Epoch {epoch+1} with Validation F1 Score: {best_val_f1}")
    return history

def save_history(history, filename):
    """
    Save the training history to a JSON file.

    Args:
    history (dict): The history dictionary containing training metrics.
    filename (str): The name of the file to save the history to.
    """
    with open(filename, 'w') as f:
        json.dump(history, f)

# Train the model
print("Training in progress...")
t_1 = time.time()
history = train_model(model, criterion, optimizer, train_loader, validation_loader, epochs)
save_history(history, 'training_history.json')

t_2 = time.time()
print("Training completed.")
print("Total training duration  = ", t_2 - t_1)

# Extract training and validation loss from history
training_loss = history['train_loss']
validation_loss = history['val_loss']
# Plot the training and validation loss

def evaluate_model(model, data_loader, classes, dataset_name="Validation"):
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
    class_report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
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
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {dataset_name} Set')
    plt.show()

# --- Section 7: Training History and Model Evaluation ---
# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(2, 1, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.plot(history['train_precision'], label='Train Precision')
plt.plot(history['train_recall'], label='Train Recall')
plt.plot(history['train_f1'], label='Train F1 Score')
plt.plot(history['val_precision'], label='Validation Precision')
plt.plot(history['val_recall'], label='Validation Recall')
plt.plot(history['val_f1'], label='Validation F1 Score')
plt.legend()
plt.title('Metrics Over Epochs')

plt.show()

print("Evaluating on validation set:")
evaluate_model(model, validation_loader, validation_dataset.classes)

print("Evaluating on test set:")
evaluate_model(model, test_loader, train_dataset.classes)

# Save the model's state dictionary
model_save_path = 'emotion_classifier.pth'  # Define your path here
torch.save(model.state_dict(), model_save_path)

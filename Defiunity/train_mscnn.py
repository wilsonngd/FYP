import os
import json
import pandas as pd
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from imblearn.over_sampling import RandomOverSampler
from hypll.optim import RiemannianAdam
from torch import nn
from hypll.tensors import TangentTensor
from hyperbolic_mscnn import HyperbolicMSCNN
from hyperbolic_mscnn_base import MSCNN
from hyperbolic_mscnn import manifold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Define paths
dataset_dir = "C:/Users/zoezh/Downloads/clothes_dataset"
train_json = os.path.join(dataset_dir, 'train/_annotations.coco.json')
val_json = os.path.join(dataset_dir, 'valid/_annotations.coco.json')
test_json = os.path.join(dataset_dir, 'test/_annotations.coco.json')

# Function to load and process COCO annotations
def load_coco_annotations(json_file, img_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']

    img_paths, labels = [], []
    annotated_img_ids = set()

    for ann in annotations:
        img_id = ann['image_id']
        label = 1
        img_path = os.path.join(img_dir, images[img_id]['file_name'])
        img_paths.append(img_path)
        labels.append(label)
        annotated_img_ids.add(img_id)

    # Process images without annotations (considered non-defect)
    for img_id, img_info in images.items():
        if img_id not in annotated_img_ids:
            img_path = os.path.join(img_dir, img_info['file_name'])
            img_paths.append(img_path)
            labels.append(0)  # Label as non-defect

    return img_paths, labels

def calculate_label_ratios(labels):
    df = pd.DataFrame(labels, columns=['label'])
    label_counts = df['label'].value_counts()
    total_images = len(labels)
    defect_count = label_counts.get(1, 0)
    non_defect_count = label_counts.get(0, 0)

    defect_ratio = defect_count / total_images
    non_defect_ratio = non_defect_count / total_images

    return defect_count, non_defect_count, defect_ratio, non_defect_ratio

# Load annotations for different sets
train_paths, train_labels = load_coco_annotations(train_json, os.path.join(dataset_dir, 'train'))
val_paths, val_labels = load_coco_annotations(val_json, os.path.join(dataset_dir, 'valid'))
test_paths, test_labels = load_coco_annotations(test_json, os.path.join(dataset_dir, 'test'))

# Calculate ratios before oversampling
print("Before Oversampling:")
train_defect_count, train_non_defect_count, train_defect_ratio, train_non_defect_ratio = calculate_label_ratios(train_labels)
val_defect_count, val_non_defect_count, val_defect_ratio, val_non_defect_ratio = calculate_label_ratios(val_labels)
test_defect_count, test_non_defect_count, test_defect_ratio, test_non_defect_ratio = calculate_label_ratios(test_labels)

print(f"Training Set - Defect Count: {train_defect_count}, Non-Defect Count: {train_non_defect_count}")
print(f"Training Set - Defect Ratio: {train_defect_ratio:.2f}, Non-Defect Ratio: {train_non_defect_ratio:.2f}")

print(f"Validation Set - Defect Count: {val_defect_count}, Non-Defect Count: {val_non_defect_count}")
print(f"Validation Set - Defect Ratio: {val_defect_ratio:.2f}, Non-Defect Ratio: {val_non_defect_ratio:.2f}")

print(f"Test Set - Defect Count: {test_defect_count}, Non-Defect Count: {test_non_defect_count}")
print(f"Test Set - Defect Ratio: {test_defect_ratio:.2f}, Non-Defect Ratio: {test_non_defect_ratio:.2f}")

# Perform oversampling
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(np.array(train_paths).reshape(-1, 1), train_labels)

# Convert back to DataFrame
train_df_resampled = pd.DataFrame({
    'filename': X_train_resampled.flatten(),
    'label': y_train_resampled
})

# Calculate ratios after oversampling
train_defect_count, train_non_defect_count, train_defect_ratio, train_non_defect_ratio = calculate_label_ratios(y_train_resampled)
print("\nAfter Oversampling:")
print(f"Training Set - Defect Count: {train_defect_count}, Non-Defect Count: {train_non_defect_count}")
print(f"Training Set - Defect Ratio: {train_defect_ratio:.2f}, Non-Defect Ratio: {train_non_defect_ratio:.2f}")

# Convert labels to strings for flow_from_dataframe
train_df_resampled['label'] = train_df_resampled['label'].astype(str)
val_df = pd.DataFrame({'filename': val_paths, 'label': val_labels}).astype(str)
test_df = pd.DataFrame({'filename': test_paths, 'label': test_labels}).astype(str)

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = plt.imread(img_name)
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = CustomDataset(train_df_resampled, os.path.join(dataset_dir, 'train'), transform=train_transform)
val_dataset = CustomDataset(val_df, os.path.join(dataset_dir, 'valid'), transform=val_test_transform)
test_dataset = CustomDataset(test_df, os.path.join(dataset_dir, 'test'), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='C:/Users/zoezh/Downloads/mscnn_gc(lr2)_.pth', verbose=False):
        self.patience = patience
        self.delta = delta 
        self.path = path  
        self.verbose = verbose 
        self.counter = 0  
        self.best_score = None 
        self.early_stop = False  

    def __call__(self, val_f1, model):
        if self.best_score is None or val_f1 > self.best_score + self.delta:
            self.best_score = val_f1  
            self.save_checkpoint(model)
            self.counter = 0  
        else:
            self.counter += 1  
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True  

    def save_checkpoint(self, model):
        """Saves the model if the validation F1 score improves"""
        if self.verbose:
            print(f"Validation F1-Score improved. Saving model...")
        torch.save(model.state_dict(), self.path) 

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = HyperbolicMSCNN(input_channels=3, num_classes=2).to(device)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = RiemannianAdam(net.parameters(), lr=0.0002, weight_decay=1e-4)
early_stopping = EarlyStopping(patience=10, verbose=True)

# Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.5, threshold=0.0001)


start_time = time.time()

total_training_time = 0
total_validation_time = 0
training_losses = []

# Training loop
for epoch in range(100):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    epoch_start_time = time.time()
    print("Starting Training Process")

    net.train()
    epoch_train_start = time.time()  # Start timing training phase

    all_train_labels = []
    all_train_predictions = []

    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold)
        manifold_inputs = manifold.expmap(tangents)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(manifold_inputs)
        loss = criterion(outputs.tensor, labels)
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.tensor, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Save predictions and labels for detailed metrics
        all_train_predictions.extend(predicted.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())
        
        # Intermediate statistics
        if i % 10 == 9:
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.5f}, accuracy: {accuracy:.9f}")
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

    # Store the average training loss for the epoch
    epoch_train_loss = running_loss / len(train_loader)  # Average loss over all batches in the epoch
    training_losses.append(epoch_train_loss)
    print(f"Epoch {epoch + 1} Training Loss: {epoch_train_loss:.4f}")

    epoch_train_end = time.time()  # End timing training phase
    total_training_time += (epoch_train_end - epoch_train_start)

    # Calculate training metrics
    train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    train_precision = precision_score(all_train_labels, all_train_predictions, average='weighted', zero_division=1)
    train_recall = recall_score(all_train_labels, all_train_predictions, average='weighted', zero_division=1)
    train_f1 = f1_score(all_train_labels, all_train_predictions, average='weighted')

    print(f"Epoch {epoch + 1} Training - Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, "
          f"Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
    
    train_cm = confusion_matrix(all_train_labels, all_train_predictions)
    print(f"Confusion Matrix for Training (Epoch {epoch + 1}):\n{train_cm}")

    # Display confusion matrix using matplotlib
    train_cm_disp = ConfusionMatrixDisplay(confusion_matrix=train_cm)
    train_cm_disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Training Epoch {epoch + 1}")
    plt.show()

    # Validation phase
    net.eval()  # Switch to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_val_labels = []
    all_val_predictions = []

    epoch_val_start = time.time()  # Start timing validation phase

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold)
            manifold_inputs = manifold.expmap(tangents)

            outputs = net(manifold_inputs)
            loss = criterion(outputs.tensor, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.tensor, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            # Save predictions and labels for detailed metrics
            all_val_predictions.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    epoch_val_end = time.time()  # End timing validation phase
    total_validation_time += (epoch_val_end - epoch_val_start)

    # Calculate validation metrics
    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    val_precision = precision_score(all_val_labels, all_val_predictions, average='weighted', zero_division=1)
    val_recall = recall_score(all_val_labels, all_val_predictions, average='weighted', zero_division=1)
    val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')

    print(f"Epoch {epoch + 1} Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
          f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

    epoch_end_time = time.time()  # End timing the epoch
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

    val_cm = confusion_matrix(all_val_labels, all_val_predictions)
    print(f"Confusion Matrix for Validation (Epoch {epoch + 1}):\n{val_cm}")

    # Display confusion matrix using matplotlib
    val_cm_disp = ConfusionMatrixDisplay(confusion_matrix=val_cm)
    val_cm_disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Validation Epoch {epoch + 1}")
    plt.show()

    scheduler.step(val_loss)

    # Early stopping
    early_stopping(val_f1, net)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

end_time = time.time()

# Calculate overall training time
overall_training_duration = end_time - start_time

# Print overall statistics
print(f"\nFinished Training in {overall_training_duration:.2f} seconds ({overall_training_duration / 60:.2f} minutes)")
print(f"Total Training Time: {total_training_time:.2f} seconds")
print(f"Total Validation Time: {total_validation_time:.2f} seconds")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Per Epoch')
plt.legend()
plt.grid(True)
plt.show()

net.load_state_dict(torch.load('C:/Users/zoezh/Downloads/mscnn_gc(lr2)_.pth'))
net = net.to(device) 
net.eval() 

all_labels = []
all_preds = []

# Testing loop
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        tangents = TangentTensor(data=images, man_dim=1, manifold=manifold)
        manifold_images = manifold.expmap(tangents)

        # calculate outputs by running images through the network
        outputs = net(manifold_images)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.tensor, 1)

        # Collect all labels and predictions for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Accuracy
accuracy = correct / total
print(f"Accuracy of the network on the test images: {accuracy} ")

# Precision, Recall, F1-Score (binary classification)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

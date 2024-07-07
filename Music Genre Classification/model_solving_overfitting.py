import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Path to the JSON file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "data.json"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenreDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as fp:
            data = json.load(fp)
        self.X = np.array(data["mfcc"])
        self.y = np.array(data["labels"])

        # Normalize the MFCCs
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def load_data(data_path):
    """Loads training dataset from json file."""
    dataset = GenreDataset(data_path)
    return dataset

class GenreClassifier(nn.Module):
    def __init__(self, input_shape):
        super(GenreClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    # Load data
    dataset = load_data(DATA_PATH)

    # Create train/test split
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build network topology
    input_shape = dataset[0][0].numel()
    model = GenreClassifier(input_shape).to(DEVICE)

    # Compile model
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Evaluate model on test data
        model.eval()
        running_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss = running_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        elapsed_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Training Loss: {train_loss:.3f}, Training Accuracy: {train_accuracy:.3f}, "
              f"Testing Loss: {test_loss:.3f}, Testing Accuracy: {test_accuracy:.3f}, "
              f"Time: {elapsed_time:.3f}s")

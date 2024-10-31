import numpy as np
import torch
import torch.nn as nn

from models.cnn.cnn import MultiMNISTDataset

class MultiLabelCNN(nn.Module):
    def __init__(self, act_func: str, lr: float, dropout: float, optimiser: str):
        super(MultiLabelCNN, self).__init__()

        self.loss = nn.BCEWithLogitsLoss()

        if act_func == 'relu':
            self.activation = torch.relu
        elif act_func == 'sigmoid':
            self.activation = torch.sigmoid
        elif act_func == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError('Activation function not supported.')

        self.lr = lr
        self.dropout = dropout
        self.op = optimiser

        # Defining Layers, Input is 128x128x1 images
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*29*29, 1024)
        self.fc2 = nn.Linear(1024, 33)

    def forward(self, x):
        x = self.maxpool1(self.activation(self.conv1(x)))
        x = self.maxpool2(self.activation(self.conv2(x)))
        x = x.view(-1, 128*29*29)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, x, y, val_x, val_y, epochs, batch_size=32):
        self.batch_size = batch_size
        torch.cuda.empty_cache()
        if self.op == 'adam':
            self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.op == 'sgd':
            self.optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError('Optimiser not supported.')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataloader = MultiMNISTDataset(x, y, batch_size=batch_size, task="regression")
        val_dataloader = MultiMNISTDataset(val_x, val_y, batch_size=batch_size, task="regression")

        train_loss_arr = list()
        val_loss_arr = list()

        train_iters = len(train_dataloader)
        val_iters = len(val_dataloader)

        for epoch in range(epochs):
            total_train_loss = 0
            total_val_loss = 0

            # Switching model to training mode
            self.train()

            for i, (images, labels) in enumerate(train_dataloader):
                train_images = images.to(device)
                train_labels = labels.to(device)

                # Forward Pass
                train_outputs = self.forward(train_images)
                train_loss = self.loss(train_outputs, train_labels)
                total_train_loss += train_loss.item()

                # Backward Pass
                self.optimiser.zero_grad()
                train_loss.backward()
                self.optimiser.step()
                print(f'Epoch {epoch+1}/{epochs}, iter [{i+1}/{train_iters}], Loss: {train_loss.item()}')
            avg_train_loss = total_train_loss / train_iters
            train_loss_arr.append(avg_train_loss)

            # Switching model to evaluation mode
            self.eval()

            # Calculating validation loss
            with torch.no_grad():
                for i, (val_images, val_labels) in enumerate(val_dataloader):
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = self.forward(val_images)
                    val_loss = self.loss(val_outputs, val_labels).item()
                    total_val_loss += val_loss
                avg_val_loss = total_val_loss / val_iters
                val_loss_arr.append(avg_val_loss)

        return train_loss_arr, val_loss_arr

    def accuracy(self, x, y, threshold=0.5):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader = MultiMNISTDataset(x, y, batch_size=self.batch_size, task="regression")

        exact_match_correct = 0
        hamming_correct = 0
        total_samples = 0
        total_labels = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device).float()  # Ensure labels are in float for multi-label

                # Forward pass
                outputs = self.forward(images)
                predictions = torch.sigmoid(outputs) > threshold  # Apply sigmoid and threshold

                # Exact Match Accuracy
                exact_match_correct += (predictions == labels).all(dim=1).sum().item()  # Count where all labels match per sample
                total_samples += labels.size(0)  # Number of samples

                # Hamming Accuracy
                hamming_correct += (predictions == labels).sum().item()  # Count matches across all labels
                total_labels += labels.numel()  # Total number of labels across all samples

        exact_match_accuracy = exact_match_correct / total_samples if total_samples > 0 else 0
        hamming_accuracy = hamming_correct / total_labels if total_labels > 0 else 0

        return exact_match_accuracy, hamming_accuracy


    def get_loss(self, x, y):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader = MultiMNISTDataset(x, y, batch_size=self.batch_size, task="regression")
        iters = len(dataloader)
        with torch.no_grad():
            total_loss = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                total_loss += self.loss(outputs, labels).item()
        return total_loss / iters
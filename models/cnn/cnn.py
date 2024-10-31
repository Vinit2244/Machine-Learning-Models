import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MultiMNISTDataset:
    def __init__(self, data, labels, batch_size=32, shuffle=True, task="classification"):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))  # Track the indexes for shuffling
        self.current_index = 0
        self.task = task

        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            self._shuffle_data()
        return self

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration

        # Compute the indices of the current batch
        start_idx = self.current_index
        end_idx = min(start_idx + self.batch_size, len(self.data))

        # Fetch the batch data and labels
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_data = self.data[batch_indexes]
        batch_labels = self.labels[batch_indexes]

        # Update the current index
        self.current_index = end_idx

        # Convert NumPy arrays to PyTorch tensors
        batch_data_tensor = torch.from_numpy(batch_data).float()
        if self.task == "classification":
            batch_labels_tensor = torch.from_numpy(batch_labels).long()
        elif self.task == "regression":
            batch_labels_tensor = torch.from_numpy(batch_labels).float()

        return batch_data_tensor.unsqueeze(1), batch_labels_tensor

class CNN(nn.Module):
    def __init__(self, task: str, act_func: str, lr: float, dropout: float, optimiser: str):
        super(CNN, self).__init__()
        self.task = task # Can be regression/classification

        if task == 'classification':
            self.loss = nn.CrossEntropyLoss()
        elif task == 'regression':
            self.loss = nn.MSELoss()

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
        self.fc2 = nn.Linear(1024, 10)
        if task == "regression":
            self.fc3 = nn.Linear(10, 1)

    def forward(self, x, return_intermediate_feature_maps=False):
        # x1 and x2 are intermediate feature maps for visualisation
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x1 = self.maxpool1(x1)

        x2 = self.conv2(x1)
        x2 = self.activation(x2)
        x2 = self.maxpool2(x2)

        x3 = x2.view(-1, 128*29*29)
        x3 = self.activation(self.fc1(x3))
        x3 = self.fc2(x3)
        if self.task == "regression":
            x3 = self.fc3(x3)

        if not return_intermediate_feature_maps:
            return x3
        else:
            return x1, x2, x3

    def visualise_feature_maps(self, x, y, save_as):
        self.eval()

        # Convert NumPy arrays to PyTorch tensors
        x = torch.from_numpy(x).float().unsqueeze(1)
        if self.task == "classification":
            y = torch.from_numpy(y).long()
        elif self.task == "regression":
            y = torch.from_numpy(y).float()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        x1, x2, _ = self.forward(x, return_intermediate_feature_maps=True)
        # x1 has 64 filters and x2 has 128 so only displaying first 64 filters of both the intermediate layers

        # Visualising layer 1 output
        for img_idx in range(x.shape[0]):
            fig, ax = plt.subplots(8, 8, figsize=(10, 10))
            for i in range(64):
                ax[i//8, i%8].imshow(x1[img_idx, i, :, :].detach().cpu().numpy(), cmap='viridis')
                ax[i//8, i%8].axis('off')
            plt.savefig(f"{save_as}_img_{img_idx}_label_{y[img_idx]}_x1.png")
            plt.close()

        # Visualising layer 2 output
        for img_idx in range(x.shape[0]):
            fig, ax = plt.subplots(8, 8, figsize=(10, 10))
            for i in range(64):
                ax[i//8, i%8].imshow(x2[img_idx, i, :, :].detach().cpu().numpy(), cmap='viridis')
                ax[i//8, i%8].axis('off')
            plt.savefig(f"{save_as}_img_{img_idx}_label_{y[img_idx]}_x2.png")
            plt.close()

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
        train_dataloader = MultiMNISTDataset(x, y, batch_size=batch_size, task=self.task)
        val_dataloader = MultiMNISTDataset(val_x, val_y, batch_size=batch_size, task=self.task)

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

    def accuracy(self, x, y):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader = MultiMNISTDataset(x, y, batch_size=self.batch_size, task=self.task)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # In case of regression I am rounding down the output value to integer as final predicted label
                if self.task == "regression":
                  predicted = torch.round(predicted.float())
                  labels = torch.round(labels)
                correct += (predicted == labels).sum().item()
        return correct / total

    def get_loss(self, x, y):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader = MultiMNISTDataset(x, y, batch_size=self.batch_size, task=self.task)
        iters = len(dataloader)
        with torch.no_grad():
            total_loss = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                total_loss += self.loss(outputs, labels).item()
        return total_loss / iters
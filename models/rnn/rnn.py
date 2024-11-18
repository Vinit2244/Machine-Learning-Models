import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Custom dataset class
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor([float(bit) for bit in self.sequences[idx]], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label
    
def function_to_collate(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sequences, labels

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_data(self, X_train, X_val, y_train, y_val, batch_size=32):
        self.batch_size = batch_size

        self.train_dataset = SequenceDataset(X_train, y_train)
        self.val_dataset = SequenceDataset(X_val, y_val)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=function_to_collate)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=function_to_collate)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        output, _ = self.rnn(x, h0)
        last_output = output[:, -1, :]
        normalized = self.layer_norm(last_output)
        prediction = self.fc(normalized)
        return prediction.squeeze()

    def train_model(self, loss="MSE", optimizer="Adam", lr=0.001, num_epochs=10):
        # Initiating Loss
        if loss == "MAE":
            self.criterion = nn.L1Loss()  # MAE
        elif loss == "MSE":
            self.criterion = nn.MSELoss()  # MSE
        elif loss == "BCE":
            self.criterion = nn.BCELoss()  # BCE

        # Initiating Optimiser
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=lr)

        train_loss_arr = list()
        val_loss_arr = list()
        mae_arr = list()

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for sequences, labels in self.train_loader:
                sequences = sequences.unsqueeze(-1)  # Add an extra dimension for input_size

                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.forward(sequences)
                loss = self.criterion(outputs.squeeze(), labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_loss_arr.append(train_loss)

            # Validation loop
            self.eval()
            val_loss = 0
            mae = 0

            with torch.no_grad():
                for sequences, labels in self.val_loader:
                    sequences = sequences.unsqueeze(-1)

                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.forward(sequences)
                    loss = self.criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

                    mae += torch.mean(torch.abs(outputs.squeeze() - labels)).item()

            val_loss /= len(self.val_loader)
            val_loss_arr.append(val_loss)
            avg_mae = mae / len(self.val_loader)
            mae_arr.append(avg_mae)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {round(train_loss, 3)}, Validation Loss: {round(val_loss, 3)}, Val MAE: {round(avg_mae, 3)}')

        return train_loss_arr, val_loss_arr, mae_arr

    def get_loss(self, X_test, y_test):
        test_dataset = SequenceDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=function_to_collate)

        self.eval()
        test_loss = 0
        mae_loss = 0
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.unsqueeze(-1)

                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.forward(sequences)
                loss = self.criterion(outputs.squeeze(), labels)
                test_loss += loss.item()

                mae_loss += torch.mean(torch.abs(outputs.squeeze() - labels)).item()

        test_loss /= len(test_loader)
        mae_loss /= len(test_loader)
        return test_loss, mae_loss
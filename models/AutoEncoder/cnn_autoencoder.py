import numpy as np
import torch
import torch.nn as nn

from models.cnn.cnn import MultiMNISTDataset

class CNN_Autoencoder(nn.Module):
    def __init__(self, model_num: int, act_func: str, lr: float, dropout: float, optimiser: str):
        super(CNN_Autoencoder, self).__init__()
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
        self.model_num = model_num

        if self.model_num==1 or self.model_num==2 or self.model_num==3:
            # Encoding layers
            self.conv1 = nn.Conv2d(1, 16, 3)                            # 1x28x28 -> 16x26x26
            self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)     # 16x26x26 -> 16x13x13
            self.conv2 = nn.Conv2d(16, 32, 3)                           # 16x13x13 -> 32x11x11
            self.maxpool2 = nn.MaxPool2d(2, 2, return_indices=True)     # 32x11x11 -> 32x5x5
            self.conv3 = nn.Conv2d(32, 64, 5)                           # 32x5x5 -> 64x1x1

            # Decoder layers
            self.inv_conv3 = nn.ConvTranspose2d(64, 32, 5)              # 64x1x1 -> 32x5x5
            self.maxunpool2 = nn.MaxUnpool2d(2, 2)                      # 32x5x5 -> 32x11x11
            self.inv_conv2 = nn.ConvTranspose2d(32, 16, 3)              # 32x11x11 -> 16x13x13
            self.maxunpool1 = nn.MaxUnpool2d(2, 2)                      # 16x13x13 -> 16x26x26
            self.inv_conv1 = nn.ConvTranspose2d(16, 1, 3)               # 16x26x26 -> 1x28x28
            self.normalise = nn.Sigmoid()                               # Normalise the output image in range [0, 1]

        elif self.model_num == 2:
            # Encoding layers
            self.conv1 = nn.Conv2d(1, 32, 5)                            # 1x28x28 -> 32x24x24
            self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)     # 32x24x24 -> 32x12x12
            self.conv2 = nn.Conv2d(32, 64, 5)                           # 64x12x12 -> 64x8x8
            self.maxpool2 = nn.MaxPool2d(2, 2, return_indices=True)     # 64x8x8 -> 64x4x4
            self.conv3 = nn.Conv2d(64, 128, 4)                          # 64x4x4 -> 128x1x1

            # Decoder layers
            self.inv_conv3 = nn.ConvTranspose2d(128, 64, 4)             # 128x1x1 -> 64x4x4
            self.maxunpool2 = nn.MaxUnpool2d(2, 2)                      # 64x4x4 -> 64x8x8
            self.inv_conv2 = nn.ConvTranspose2d(64, 32, 5)              # 64x8x8 -> 32x12x12
            self.maxunpool1 = nn.MaxUnpool2d(2, 2)                      # 32x12x12 -> 32x24x24
            self.inv_conv1 = nn.ConvTranspose2d(32, 1, 5)               # 32x24x24 -> 1x28x28
            self.normalise = nn.Sigmoid()                               # Normalise the output image in range [0, 1]

        elif self.model_num == 3:
            # Encoding layers
            self.conv1 = nn.Conv2d(1, 64, 5)                            # 1x28x28 -> 64x24x24
            self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)     # 64x24x24 -> 64x12x12
            self.conv2 = nn.Conv2d(64, 128, 5)                          # 64x12x12 -> 128x8x8
            self.maxpool2 = nn.MaxPool2d(2, 2, return_indices=True)     # 128x8x8 -> 128x4x4
            self.conv3 = nn.Conv2d(128, 256, 4)                         # 128x4x4 -> 256x1x1

            # Decoder layers
            self.inv_conv3 = nn.ConvTranspose2d(256, 128, 4)            # 256x1x1 -> 128x4x4
            self.maxunpool2 = nn.MaxUnpool2d(2, 2)                      # 128x4x4 -> 128x8x8
            self.inv_conv2 = nn.ConvTranspose2d(128, 64, 5)             # 128x8x8 -> 64x12x12
            self.maxunpool1 = nn.MaxUnpool2d(2, 2)                      # 64x12x12 -> 64x24x24
            self.inv_conv1 = nn.ConvTranspose2d(64, 1, 5)               # 64x24x24 -> 1x28x28
            self.normalise = nn.Sigmoid()                               # Normalise the output image in range [0, 1]

        # Keeping the latent space dimensions same and varying the depth of the model
        elif self.model_num == 4:
            # Encoding layers
            self.conv1 = nn.Conv2d(1, 256, 5)                           # 1x28x28 -> 64x24x24
            self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)     # 64x24x24 -> 64x12x12
            self.conv2 = nn.Conv2d(256, 128, 12)                        # 64x12x12 -> 128x1x1

            # Decoder layers
            self.inv_conv2 = nn.ConvTranspose2d(128, 256, 12)           # 128x1x1 -> 64x12x12
            self.maxunpool1 = nn.MaxUnpool2d(2, 2)                      # 64x12x12 -> 64x24x24
            self.inv_conv1 = nn.ConvTranspose2d(256, 1, 5)              # 64x24x24 -> 1x28x28
            self.normalise = nn.Sigmoid()                               # Normalise the output image in range [0, 1]

        elif self.model_num == 5:
            # Encoding layers
            self.conv1 = nn.Conv2d(1, 512, 5)                           # 1x28x28 -> 512x24x24
            self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)     # 512x24x24 -> 512x12x12
            self.conv2 = nn.Conv2d(512, 256, 5)                         # 512x12x12 -> 256x8x8
            self.maxpool2 = nn.MaxPool2d(2, 2, return_indices=True)    # 256x8x8 -> 256x4x4
            self.conv3 = nn.Conv2d(256, 128, 4)                         # 256x4x4 -> 128x1x1

            # Decoder layers
            self.inv_conv3 = nn.ConvTranspose2d(128, 256, 4)            # 128x1x1 -> 256x4x4
            self.maxunpool2 = nn.MaxUnpool2d(2, 2)                      # 256x4x4 -> 256x8x8
            self.inv_conv2 = nn.ConvTranspose2d(256, 512, 5)            # 256x8x8 -> 512x12x12
            self.maxunpool1 = nn.MaxUnpool2d(2, 2)                      # 512x12x12 -> 512x24x24
            self.inv_conv1 = nn.ConvTranspose2d(512, 1, 5)              # 512x24x24 -> 1x28x28
            self.normalise = nn.Sigmoid()                               # Normalise the output image in range [0, 1]

        elif self.model_num == 6:
            # Encoding layers
            self.conv1 = nn.Conv2d(1, 1024, 3)                          # 1x28x28 -> 1024x26x26
            self.maxpool1 = nn.MaxPool2d(2, 2, return_indices=True)     # 1024x26x26 -> 1024x13x13
            self.conv2 = nn.Conv2d(1024, 512, 3)                        # 1024x13x13 -> 512x11x11
            self.maxpool2 = nn.MaxPool2d(2, 2, return_indices=True)     # 512x11x11 -> 512x5x5
            self.conv3 = nn.Conv2d(512, 256, 3)                         # 512x5x5 -> 386x3x3
            self.conv4 = nn.Conv2d(256, 128, 3)                         # 386x3x3 -> 256x1x1

            # Decoding layers
            self.inv_conv4 = nn.ConvTranspose2d(128, 256, 3)
            self.inv_conv3 = nn.ConvTranspose2d(256, 512, 3)
            self.maxunpool2 = nn.MaxUnpool2d(2, 2)
            self.inv_conv2 = nn.ConvTranspose2d(512, 1024, 3)
            self.maxunpool1 = nn.MaxUnpool2d(2, 2)
            self.inv_conv1 = nn.ConvTranspose2d(1024, 1, 3)
            self.normalise = nn.Sigmoid()                               # Normalise the output image in range [0, 1]

    def encoder(self, x):
        if self.model_num==1 or self.model_num==2 or self.model_num==3:
            # Initialising values to None
            indices1 = indices2 = input_shape1 = input_shape2 = None

            x = self.activation(self.conv1(x))
            input_shape1 = x.shape
            x, indices1 = self.maxpool1(x)
            x = self.activation(self.conv2(x))
            input_shape2 = x.shape
            x, indices2 = self.maxpool2(x)
            x = self.activation(self.conv3(x))
            return x, indices1, indices2, input_shape1, input_shape2

        elif self.model_num==4:
            indices1 = input_shape1 = None
            x = self.activation(self.conv1(x))
            input_shape1 = x.shape
            x, indices1 = self.maxpool1(x)
            x = self.activation(self.conv2(x))
            return x, indices1, input_shape1

        elif self.model_num==5:
            indices1 = indices2 = input_shape1 = input_shape2 = None
            x = self.activation(self.conv1(x))
            input_shape1 = x.shape
            x, indices1 = self.maxpool1(x)
            x = self.activation(self.conv2(x))
            input_shape2 = x.shape
            x, indices2 = self.maxpool2(x)
            x = self.activation(self.conv3(x))
            return x, indices1, indices2, input_shape1, input_shape2

        elif self.model_num==6:
            indices1 = indices2 = input_shape1 = input_shape2 = 0
            x = self.activation(self.conv1(x))
            input_shape1 = x.shape
            x, indices1 = self.maxpool1(x)
            x = self.activation(self.conv2(x))
            input_shape2 = x.shape
            x, indices2 = self.maxpool2(x)
            x = self.activation(self.conv3(x))
            x = self.activation(self.conv4(x))
            return x, indices1, indices2, input_shape1, input_shape2

    def decoder(self, x, indices1=None, indices2=None, input_shape1=None, input_shape2=None):
        if self.model_num==1 or self.model_num==2 or self.model_num==3:
            x = self.activation(self.inv_conv3(x))
            x = self.maxunpool2(x, indices2, output_size=input_shape2)
            x = self.activation(self.inv_conv2(x))
            x = self.maxunpool1(x, indices1, output_size=input_shape1)
            x = self.activation(self.inv_conv1(x))
            x = self.normalise(x)

        elif self.model_num==4:
            x = self.activation(self.inv_conv2(x))
            x = self.maxunpool1(x, indices1, output_size=input_shape1)
            x = self.activation(self.inv_conv1(x))
            x = self.normalise(x)

        elif self.model_num==5:
            x = self.activation(self.inv_conv3(x))
            x = self.maxunpool2(x, indices2, output_size=input_shape2)
            x = self.activation(self.inv_conv2(x))
            x = self.maxunpool1(x, indices1, output_size=input_shape1)
            x = self.activation(self.inv_conv1(x))
            x = self.normalise(x)

        elif self.model_num==6:
            x = self.activation(self.inv_conv4(x))
            x = self.activation(self.inv_conv3(x))
            x = self.maxunpool2(x, indices2, output_size=input_shape2)
            x = self.activation(self.inv_conv2(x))
            x = self.maxunpool1(x, indices1, output_size=input_shape1)
            x = self.activation(self.inv_conv1(x))
            x = self.normalise(x)

        return x

    def forward(self, x):
        if self.model_num==1 or self.model_num==2 or self.model_num==3 or self.model_num==5 or self.model_num==6:
            encoded, indices1, indices2, input_shape1, input_shape2 = self.encoder(x)
            decoded = self.decoder(encoded, indices1, indices2, input_shape1, input_shape2)

        elif self.model_num==4:
            encoded, indices1, input_shape1 = self.encoder(x)
            decoded = self.decoder(x=encoded, indices1=indices1, input_shape1 = input_shape1)

        return decoded

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
                train_loss = self.loss(train_outputs, train_labels.unsqueeze(1))
                total_train_loss += train_loss.item()

                # Backward Pass
                self.optimiser.zero_grad()
                train_loss.backward()
                self.optimiser.step()

            avg_train_loss = total_train_loss / train_iters
            train_loss_arr.append(avg_train_loss)
            print(f'Epoch {epoch+1}/{epochs} done, Train Loss: {avg_train_loss}')

            # Switching model to evaluation mode
            self.eval()

            # Calculating validation loss
            with torch.no_grad():
                for i, (val_images, val_labels) in enumerate(val_dataloader):
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    val_outputs = self.forward(val_images)
                    val_loss = self.loss(val_outputs, val_labels.unsqueeze(1)).item()
                    total_val_loss += val_loss
                avg_val_loss = total_val_loss / val_iters
                val_loss_arr.append(avg_val_loss)

        return train_loss_arr, val_loss_arr

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
                total_loss += self.loss(outputs, labels.unsqueeze(1)).item()
        return total_loss / iters

    def get_encoded_representation(self, x, y):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader = MultiMNISTDataset(x, y, batch_size=self.batch_size, shuffle=False, task="regression")
        iters = len(dataloader)
        with torch.no_grad():
            encoded_reps = list()
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                if self.model_num==1 or self.model_num==2 or self.model_num==3 or self.model_num==5 or self.model_num==6:
                    encoded, indices1, indices2, input_shape1, input_shape2 = self.encoder(images)
                elif self.model_num==4:
                    encoded, indices1, input_shape1 = self.encoder(images)
                for latent_image in encoded:
                    encoded_reps.append(latent_image.detach().cpu().numpy())
        return np.array(encoded_reps)

    def get_reconstructed_images(self, x, y):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataloader = MultiMNISTDataset(x, y, batch_size=self.batch_size, shuffle=False, task="regression")
        iters = len(dataloader)
        with torch.no_grad():
            reconstructed_images = list()
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.forward(images)
                for image in outputs:
                    reconstructed_images.append(image.detach().cpu().numpy())
        return np.array(reconstructed_images).squeeze(1)
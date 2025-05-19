import numpy as np
import torch.nn as nn
from torch.optim import Adam

class DLModel(nn.Module):

    def __init__(self, input_size, optimizer = 'Adam', conv1_size = 64, conv2_size = 128, fc_size = 512):
        max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.model = nn.Sequential(
            nn.Conv2d(input_size, out_channels=64, kernel_size=4),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4),
            max_pool,
            nn.Conv2d(32, out_channels=128, kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            max_pool,
            nn.Linear(64, fc_size),
            nn.Linear(fc_size, fc_size),
            nn.Linear(fc_size, 2)
        )

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters())

    def fit(self):
        pass

    def train(self, train_loader):
        loss = 0
        accuracy = 0

        self.model.train()
        for x, y in train_loader:
            output = self.model(x)
            self.optimizer.zero_grad()
            batch_loss = self.loss_function(output, y)
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()
            #accuracy += get_batch_accuracy(output, y, train_N)
        print('TRAIN - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

    def predict(self, image: np.ndarray) -> np.ndarray:
        # inne dane wejściowe
        return self.model.predict(image)

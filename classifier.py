import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=10),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, 8, 0),

            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=10),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, 8, 0),

            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=10),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.MaxPool2d(8, 8, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(216, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 2]
        # Extract features by convolutional layers.
        x = self.cnn_layers(x)
        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)
        #print('xshape: ', x.shape)
        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

class LogisticClassifier(nn.Module):
    def __init__(self):
        super(LogisticClassifier, self).__init__()
        self.linear_layer = nn.Linear(224*224*3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(1)
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        #print(x)
        return x.flatten(0)

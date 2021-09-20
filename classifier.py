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
            nn.Linear(216, 32),
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

"""class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(8, 8)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16)
        #print('x shape: ', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x"""
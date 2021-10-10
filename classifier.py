import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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
        x = self.cnn_layers(x)
        #Flatten the tensor to a vector so it can be input to linear layers
        x = x.flatten(1)
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
        return x.flatten(0)


class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.linear_layer = nn.Linear(224*224*3, 1)

    def forward(self, x):
        x = x.flatten(1)
        x = self.linear_layer(x)
        return x.flatten(0)

class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
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
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x
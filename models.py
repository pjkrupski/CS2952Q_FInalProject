# Define all model architectures here.
import torch.nn as nn

class CNNModel_640(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
          nn.Conv2d(1, 8, 3, 3), # in_channels, out_channels, kernel_size, stride
          nn.ReLU(),
          nn.Conv2d(8, 16, 3, 3),
          nn.ReLU(),
          nn.Conv2d(16, 32, 3, 3),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout(0.25),
          nn.Flatten(),
          nn.Linear(3872, 128), #in_features, out_features
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(128, 10),
          nn.Softmax(dim=-1)
        )

    def forward(self, input):
      return self.model(input)

class CNNModel_128(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
          nn.Conv2d(1, 96, 8, 2), # in_channels, out_channels, kernel_size, stride
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(96, 256, 6, 2),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(256, 384, 3, 1),
          nn.ReLU(),
          # nn.MaxPool2d(2),
          nn.Dropout(0.5),
          nn.Flatten(),
          nn.Linear(6144, 512), #in_features, out_features
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(512, 128), #in_features, out_features
          nn.ReLU(),
          nn.Linear(128, 10),
          nn.Softmax(dim=-1)
        )

    def forward(self, input):
      return self.model(input)

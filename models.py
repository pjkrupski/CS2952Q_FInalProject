# Define all model architectures here.
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
          nn.Conv2d(1, 8, 3, 1),
          nn.ReLU(),
          nn.Conv2d(8, 16, 3, 1),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Dropout(0.25),
          nn.Flatten(),
          nn.Linear(2304, 128),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(128, 10),
          nn.Softmax(dim=-1)
        )

    def forward(self, input):
      return self.model(input)

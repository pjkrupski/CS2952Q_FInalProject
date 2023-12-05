# Define all model architectures here.
import torch.nn as nn
from vit_pytorch import SimpleViT, ViT
from vit_pytorch.deepvit import DeepViT
import torch.nn.functional as F
import torch

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

def softmax_temp(input, t=1.0):
   ex = torch.exp(input/t)
   sum = torch.sum(ex, axis=0)
   return ex/sum

class CNNModel_128(nn.Module):
    def __init__(self, is_teacher = False, temp=1):
        super().__init__()

        self.teacher = is_teacher
        self.temp = temp
        self.model = nn.Sequential(
          nn.Conv2d(1, 96, 8, 2), # in_channels, out_channels, kernel_size, stride
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(96, 256, 6, 2),
          nn.ReLU(),
          nn.MaxPool2d(2),
          nn.Conv2d(256, 384, 3, 1),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Flatten(),
          nn.Linear(6144, 512), #in_features, out_features
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(512, 128), #in_features, out_features
          nn.ReLU(),
          nn.Linear(128, 10),
        )

    def forward(self, input):
      output = self.model(input)
      if self.is_teacher or self.training:
        return softmax_temp(output, 1.2)
      else:
        return softmax_temp(output, 1)


VitModel = DeepViT(
   image_size=128,
   patch_size = 16,
   num_classes = 10,
   dim=256,
   depth=6,
   heads=16,
   mlp_dim=756,
   dropout=0.25,
   emb_dropout=0.25,
   channels=1
)
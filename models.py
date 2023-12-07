# Define all model architectures here.
import torch.nn as nn
from vit_pytorch import SimpleViT, ViT
from vit_pytorch.deepvit import DeepViT
import torch.nn.functional as F
import torch

def softmax_temp(input, t=1.0):
    maxes = torch.max(input/t, 1, keepdim=True)[0]
    ex = torch.exp((input)/t - maxes)
    sum = torch.sum(ex, axis=1, keepdim=True)
    return ex/sum

class CNNModel_128(nn.Module):
    def __init__(self, is_teacher = False, temp=1):
        super().__init__()

        self.is_teacher = is_teacher
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
        return softmax_temp(output, self.temp)
      else:
        return softmax_temp(output, 1)


VitModel = ViT(
   image_size=128,
   patch_size = 16,
   num_classes = 10,
   dim=256,
   depth=6,
   heads=12,
   mlp_dim=512,
   dropout=0.1,
   emb_dropout=0.1,
   channels=1
)

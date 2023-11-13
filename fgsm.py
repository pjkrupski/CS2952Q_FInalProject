#Import images
#Apply FGSM pertubation
#Save altered images

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from absl import flags

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

from models import CNNModel_128
from preprocess import load_single_data, load_single_labels, load_single_image

pretrained_model = "model.pt"
use_cuda=True
# Set random seed for reproducibility
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else "cpu"
model = CNNModel_128().to(device)
model.load_state_dict(torch.load(pretrained_model, map_location=device))
model.eval()

#No epsilon will exceed 1

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


# Doc https://abseil.io/docs/python/guides/flags
FLAGS = flags.FLAGS

train_loader, test_loader = load_single_data(64)
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    #FLAGS.eps throws attribute error 
    perturbed_image = fast_gradient_method(model, data, .3, np.inf)
    print(io.decode_image(perturbed_image))
    break 


       




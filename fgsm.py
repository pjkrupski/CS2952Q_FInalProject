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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from absl import flags

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from models import CNNModel_128
from preprocess import load_single_data, load_single_labels, load_single_image

pretrained_model = "128b_120e.pt"
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

train_loader, test_loader = load_single_data(4)
for data, target, filename in test_loader:
    #data, target = data.to(device), target.to(device)
    #squashed_data = torch.squeeze(data, dim=0)
    #model(data)
    #perturbed_image = projected_gradient_descent(model, data, .001, .001, 100, np.inf)
    perturbed_image = fast_gradient_method(model, data, .001, np.inf)

    to_pil = transforms.ToPILImage()
    image = to_pil(perturbed_image[0])
    image.show()
    image.save(f"{filename}.jpg")
    
    break 


       
    """for d in data:
        perturbed_image = fast_gradient_method(model, d, .3, np.inf)
        to_pil = transforms.ToPILImage()
        image = to_pil(d)
        image.show()
        image.save("perturbed.jpg")
        #print(perturbed_image)
        break
    """



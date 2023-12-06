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
from tqdm import tqdm

from absl import flags

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from models import CNNModel_128, VitModel
from preprocess import load_single_data, load_single_labels, load_single_image


def iterative_attack(model, attack_model, device, loader):
    # Attack the model up to epsilon = 0.5, theoretical max.
    accs = []
    #for i in tqdm(range(3, 40)):
    #    i /= 3
    #    eps = round(i / 100.0, 3)
    #PGD arg must match eps min

    #Range updated from 37 EPSs between (.01-.13) -> 26 EPSs between (.001-.009) 
    #Cut in thirds for vit training (3, 12) (12, 21) (21, 29)
    for i in tqdm(range(3, 29)):
        i /= 30
        eps = round(i / 100.0, 4)
        acc = attack_test_fgsm(model, attack_model, device, loader, eps)
        accs.append((eps, acc))
    return accs

def attack_test_pgd(model, device, test_loader, eps):
    correct = 0
    total = 0
    for data, target, filenames in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data_perturbed = projected_gradient_descent(model, data, eps, 0.001, 100, np.inf)
        _, pred = model(data_perturbed).max(1)
        correct += pred.eq(torch.argmax(target, dim=1)).sum().item()
        total += pred.size(0)
    return correct / total

def attack_test_fgsm(model, attack_model, device, test_loader, eps):
    correct = 0
    total = 0
    for data, target, filenames in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data_perturbed = fast_gradient_method(attack_model, data, eps, np.inf)
        _, pred = model(data_perturbed).max(1)
        correct += pred.eq(torch.argmax(target, dim=1)).sum().item()
        total += pred.size(0)
    return correct / total
  
def gen_examples(model, device, test_loader, eps):
  for data, target, filenames in test_loader:
    data, target = data.to(device), target.to(device)
    data_perturbed = fast_gradient_method(model, data, eps, np.inf)
    to_pil = transforms.ToPILImage()
    image = to_pil(data_perturbed[0])
    #image.show()
    image.save(f"{filenames[0]}_perturbed.jpg")

def main():
  pretrained_model = "vit_none.pt"
  perturbation_calc_model = "128b_120e.pt"

  # Set random seed for reproducibility
  torch.manual_seed(42)

  device = 'cuda' if torch.cuda.is_available() else "cpu"
  model = VitModel.to(device)
  attack_model = CNNModel_128().to(device)
  #model = CNNModel_128().to(device)
  model.load_state_dict(torch.load(pretrained_model, map_location=device))
  attack_model.load_state_dict(torch.load(perturbation_calc_model, map_location=device))
  attack_model.eval()
  model.eval()

  _, test_loader = load_single_data(4, True) #True if ViT

  # gen_examples(model, device, test_loader, 0.001)
  accs = iterative_attack(model, attack_model, device, test_loader)
  file = open('./attack_results.txt', 'w')
  file.write(str(accs))
  file.close()

if __name__ == '__main__':
  main()
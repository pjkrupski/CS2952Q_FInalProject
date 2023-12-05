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

from models import CNNModel_128, CNNModel_640, VitModel
from preprocess import load_single_data, load_single_labels, load_single_image


def iterative_attack(model, device, loader):
    # Attack the model up to epsilon = 0.5, theoretical max.
    accs = []
    for i in tqdm(range(0, 51)):
        eps = i / 100.0
        acc = attack_test_pgd(model, device, loader, eps)
        accs.append((eps, acc))
    return accs

def attack_test_pgd(model, device, test_loader, eps):
    correct = 0
    total = 0
    for data, target, filenames in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data_perturbed = projected_gradient_descent(model, data, eps, 0.01, 100, np.inf)
        _, pred = model(data_perturbed).max(1)
        correct += pred.eq(torch.argmax(target, dim=1)).sum().item()
        total += pred.size(0)
    return correct / total

def attack_test_fgsm(model, device, test_loader, eps):
    correct = 0
    total = 0
    for data, target, filenames in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        data_perturbed = fast_gradient_method(model, data, eps, np.inf)
        _, pred = model(data_perturbed).max(1)
        correct += pred.eq(torch.argmax(target, dim=1)).sum().item()
        total += pred.size(0)
    return correct / total
  
def gen_examples(model, device, test_loader, eps):
  for data, target, filenames in test_loader:
    data, target = data.to(device), target.to(device)
    # data_perturbed = fast_gradient_method(model, data, eps, np.inf)
    data_perturbed = projected_gradient_descent(model, data, eps, 0.01, 100, np.inf)
    print(data.shape, data_perturbed.shape)
    plt.imshow(data_perturbed[0].permute(1, 2, 0).cpu().detach().numpy())
    plt.show()
    plt.imshow(data[0].permute(1, 2, 0).cpu().detach().numpy())
    plt.show()
    # plt.savefig(filenames[0])
    # to_pil = transforms.ToPILImage()
    # image = to_pil(data_perturbed[0])
    # print(image)
    # image.show()
    # image.save(f"{filenames[0]}.jpg")

def main():
  pretrained_model = "vit_5percent_0.03eps_fgsm.pt"

  # Set random seed for reproducibility
  torch.manual_seed(42)

  device = 'cuda' if torch.cuda.is_available() else "cpu"
  model = VitModel.to(device)
  model.load_state_dict(torch.load(pretrained_model, map_location=device))
  model.eval()

  _, test_loader = load_single_data(4)

  gen_examples(model, device, test_loader, 0.01)
  # accs = iterative_attack(model, device, test_loader)
  # file = open('./attack_results.txt', 'w')
  # file.write(str(accs))
  # file.close()

if __name__ == '__main__':
  main()
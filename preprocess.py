import os
from typing import List
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import csv
import numpy as np
from PIL import Image

def load_labels(filepath):
  image_names = []
  labels = []
  with open(filepath, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    headers = next(reader)
    for row in reader:
      image_names.append(row[0])
      onehot = list(map(lambda x: int(x), row[1:]))
      labels.append(torch.IntTensor(onehot))
  
  return image_names, labels

def load_data(batch_size=16):
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  train_data = UATD_Dataset('./data/train', './data/train/_classes.csv', transform)
  test_data = UATD_Dataset('./data/test', './data/test/_classes.csv', transform)
  train_loader = DataLoader(train_data, batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size, shuffle=True)
  return train_loader, test_loader

class UATD_Dataset(Dataset):
  def __init__(self, image_dir, labels_file, transform=None):
    self.image_dir = image_dir
    self.image_names, self.labels = load_labels(labels_file)
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    label = self.labels[idx]
    # image = read_image(os.path.join(self.image_dir, self.image_names[idx]))
    image = Image.open(os.path.join(self.image_dir, self.image_names[idx]))
    image = image.convert('L')
    if self.transform:
      image = self.transform(image)
    return image, label
import os
from typing import List
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import csv
import numpy as np
from PIL import Image

def load_multi_data(batch_size=16):
  transform = transforms.Compose([transforms.ToTensor()])
  train_data = UATD_Multi_Dataset('./data/train', './data/train/_classes.csv', transform)
  test_data = UATD_Multi_Dataset('./data/test', './data/test/_classes.csv', transform)
  train_loader = DataLoader(train_data, batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size, shuffle=True)
  return train_loader, test_loader

def load_multi_labels(filepath):
  image_names = []
  labels = []
  with open(filepath, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    headers = next(reader)
    for row in reader:
      image_names.append(row[0])
      onehot = list(map(lambda x: float(x), row[1:]))
      labels.append(torch.FloatTensor(onehot))
  
  return image_names, labels

class UATD_Multi_Dataset(Dataset):
  def __init__(self, image_dir, labels_file, transform=None):
    self.image_dir = image_dir
    self.image_names, self.labels = load_multi_labels(labels_file)
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    label = self.labels[idx]
    image = Image.open(os.path.join(self.image_dir, self.image_names[idx]))
    image = image.convert('L')
    if self.transform:
      image = self.transform(image)
    return image, label

def load_single_data(batch_size=16):
  transform = transforms.Compose([transforms.ToTensor()])
  train_data = UATD_Single_Dataset('./data/train', './data/train/_annotations.csv', transform)
  test_data = UATD_Single_Dataset('./data/test', './data/test/_annotations.csv', transform)
  train_loader = DataLoader(train_data, batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size, shuffle=True)
  return train_loader, test_loader

def load_single_labels(filepath):
  classes = {
    'ball': 0,
    'circle cage': 1,
    'cube': 2,
    'cylinder': 3,
    'human body': 4,
    'metal bucket': 5,
    'plane': 6,
    'rov': 7,
    'square cage': 8,
    'tyre': 9
  }

  image_names = []
  labels = []
  bbox = []
  with open(filepath, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    headers = next(reader)
    for row in reader:
      if len(row) == 0:
        continue
      image_names.append(row[0])
      onehot = torch.nn.functional.one_hot(torch.tensor(classes[row[3]], dtype=torch.int64), 10)
      labels.append(onehot.type(torch.FloatTensor))
      bounds = list(map(lambda x: int(x), row[4:8]))
      bbox.append(torch.IntTensor(bounds))
  
  return image_names, labels, bbox

def load_single_image(filepath, bbox):
  image = Image.open(filepath)
  image = image.convert('L')
  center = (int(bbox[2]-bbox[0])/2, int(bbox[3] - bbox[1])/2) # x, y

  # Bounds adjustments
  left = center[0] - 64
  if left < 0:
    left = 0
  elif left + 128 > 639:
    left = 639 - 128

  top = center[1] - 64
  if top < 0:
    top = 0
  elif top + 128 > 639:
    top = 639 - 128
  image = image.crop((left, top, left + 128, top + 128)) # left, top, right, bottom
  return image

class UATD_Single_Dataset(Dataset):
  def __init__(self, image_dir, labels_file, transform=None):
    self.image_dir = image_dir
    self.image_names, self.labels, self.bboxes = load_single_labels(labels_file)
    self.transform = transform

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    label = self.labels[idx]
    image = load_single_image(os.path.join(self.image_dir, self.image_names[idx]), self.bboxes[idx])
    if self.transform:
      image = self.transform(image)
    return image, label
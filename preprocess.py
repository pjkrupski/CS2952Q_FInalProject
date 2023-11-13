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
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = UATD_Single_Dataset('./data/train', './data/train/_annotations.csv', transform)
    test_data = UATD_Single_Dataset('./data/test', './data/test/_annotations.csv', transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

# Returns balanced data set with format[{image_file, label, bbox}...]
def load_single_labels(filepath, is_test=False):
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

    splits = {
      'ball': [],
      'circle cage': [],
      'cube': [],
      'cylinder': [],
      'human body': [],
      'metal bucket': [],
      'plane': [],
      'rov': [],
      'square cage': [],
      'tyre': [] 
    }

    bbox = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        headers = next(reader)
        for row in reader:
            if len(row) == 0:
                continue
            image_name = row[0]
            onehot = torch.nn.functional.one_hot(torch.tensor(classes[row[3]], dtype=torch.int64), 10)
            label = onehot.type(torch.FloatTensor)
            bounds = list(map(lambda x: int(x), row[4:8]))
            bbox = torch.IntTensor(bounds)

            splits[row[3]].append((image_name, label, bbox))
    
    # Ensure number of examples per type is the same unless testing.
    entries = []
    min_count = min(list(map(lambda s: len(splits[s]), splits)))
    for s in splits:
        if not is_test:
          splits[s] = splits[s][:min_count]
        entries.extend(splits[s])
    return entries

def load_single_image(filepath, bbox):
    image = Image.open(filepath)
    image = image.convert('L')
    # image.show()
    center = (int(bbox[2] + bbox[0])/2, int(bbox[3] + bbox[1])/2) # x, y

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
    #image.show()
    return image

class UATD_Single_Dataset(Dataset):
  def __init__(self, image_dir, labels_file, transform=None):
    self.image_dir = image_dir
    self.entries = load_single_labels(labels_file)
    self.transform = transform

  def __len__(self):
    return len(self.entries)

  def __getitem__(self, idx):
    label = self.entries[idx][1]
    image = load_single_image(os.path.join(self.image_dir, self.entries[idx][0]), self.entries[idx][2])
    if self.transform:
      image = self.transform(image)
    return image, label

def test():
    entries = load_single_labels('./data/train/_annotations.csv')
    for e in entries:
      filepath = os.path.join('./data/train', e[0])
      image = load_single_image(filepath, e[2])

if __name__ == "__main__":
   test()
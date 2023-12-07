import os
from typing import List
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

def normalize_epsilon(data_loader):
  maxes = []
  for data, target, filename in tqdm(data_loader):
      # Find max pixel intensities for each image in batch.
      batch_maxes = torch.amax(data, dim=(1, 2, 3))
      maxes.append(batch_maxes)
  maxes = torch.concat(maxes)
  mean = torch.mean(maxes).item()
  return mean/2

def normalize_data(loader):
  means = []
  stds = []
  for data, label, filename in loader:
      means.append(torch.mean(data))
      stds.append(torch.std(data))
  
  mean = torch.mean(torch.tensor(means))
  std = torch.mean(torch.tensor(stds))
  return mean, std

def load_single_data(batch_size=16, augment = False, teacher = None):
    soft_labels = None
    if teacher is not None:
      train_data = UATD_Single_Dataset('./data/train', './data/train/_annotations.csv', transforms.Compose([transforms.ToTensor()]))
      train_loader = DataLoader(train_data, batch_size)
      soft_labels = {}
      device = 'cuda' if torch.cuda.is_available() else "cpu"
      print("Generating soft labels")
      for data, label, filename in tqdm(train_loader):
        data = data.to(device)
        soft_label = teacher(data)
        for i in range(soft_label.size()[0]):
          soft_labels[filename[i]] = soft_label.data[i]
      del train_loader
      del train_data

    transform = None
    if augment:
      transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(1, 1.1)), transforms.ToTensor()])
    else:
      transform = transforms.Compose([transforms.ToTensor()])

    train_data = UATD_Single_Dataset('./data/train', './data/train/_annotations.csv', transform, labels=soft_labels)
    test_data = UATD_Single_Dataset('./data/test', './data/test/_annotations.csv', transform, is_test=True)
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
    # min_count = min(list(map(lambda s: len(splits[s]), splits)))
    for s in splits:
        # if not is_test:
        #   splits[s] = splits[s][:min_count]
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
    # image.show()
    return image

class UATD_Single_Dataset(Dataset):
  def __init__(self, image_dir, labels_file, transform=None, is_test=False, labels=None):
    self.image_dir = image_dir
    self.entries = load_single_labels(labels_file, is_test)
    self.transform = transform
    self.labels = labels

  def __len__(self):
    return len(self.entries)

  def __getitem__(self, idx):
    filename = self.entries[idx][0]
    label = None
    if self.labels is not None:
       label = self.labels[filename]
    else:
      label = self.entries[idx][1]
    image = load_single_image(os.path.join(self.image_dir, filename), self.entries[idx][2])
    if self.transform:
      image = self.transform(image)
    return image, label, filename

if __name__ == "__main__":
    train_loader, test_loader = load_single_data(16)
    # mean, std = normalize_data(train_loader)
    # print(mean, std)
    max_epsilon = normalize_epsilon(test_loader)
    print(max_epsilon)
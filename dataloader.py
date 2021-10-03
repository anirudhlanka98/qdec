import csv
import torch
from torch.utils.data import random_split
import numpy as np


def dataloader(filepath, device):

  with open(filepath) as f:
    reader = csv.reader(f)
    data = list(reader)

  sample_size = len(data)
  non_test_size = int(0.8 * sample_size)
  test_size = sample_size - non_test_size
  train_size = int(0.8 * non_test_size)
  valid_size = non_test_size - train_size

  non_test_set, test_set = random_split(data, [non_test_size, test_size], generator=torch.Generator().manual_seed(428))
  train_set, valid_set = random_split(non_test_set, [train_size, valid_size], generator=torch.Generator().manual_seed(1000))

  train_syndromes, train_error_labels, valid_syndromes, valid_error_labels, test_syndromes, test_error_labels = ([] for _ in range(6) )

  for row in range(train_size):
    train_syndromes.append([int(x) for x in train_set[row][0][1:-1].split(',')])
    train_error_labels.append([int(x) for x in train_set[row][1][1:-1].split(',')])

  for row in range(valid_size):
    valid_syndromes.append([int(x) for x in valid_set[row][0][1:-1].split(',')])
    valid_error_labels.append([int(x) for x in valid_set[row][1][1:-1].split(',')])

  for row in range(test_size):
    test_syndromes.append([int(x) for x in test_set[row][0][1:-1].split(',')])
    test_error_labels.append([int(x) for x in test_set[row][1][1:-1].split(',')])

  train_syndromes = torch.tensor(np.array(train_syndromes), dtype=torch.float, device=device)
  train_error_labels = torch.tensor(np.array(train_error_labels), dtype=torch.float, device=device)
  valid_syndromes = torch.tensor(np.array(valid_syndromes), dtype=torch.float, device=device)
  valid_error_labels = torch.tensor(np.array(valid_error_labels), dtype=torch.float, device=device)
  test_syndromes = torch.tensor(np.array(test_syndromes), dtype=torch.float, device=device)
  test_error_labels = torch.tensor(np.array(test_error_labels), dtype=torch.float, device=device)

  return train_syndromes, train_error_labels, valid_syndromes, valid_error_labels, test_syndromes, test_error_labels
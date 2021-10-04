import dataloader as dl
from neural_net import Net, train
import sys
import datetime
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn

# Activation functions
elu = nn.ELU # Exponential linear function
softmax = nn.Softmax(dim=0) # softmax(x_i) = \exp(x_i) / (\sum_j \exp(x_j))
tanh = nn.Tanh()
relu = nn.ReLU()
sigmoid = nn.Sigmoid()

# Stabilizers and logical operators (X & Z)
steane_stabs = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]) # (6 x 14)

#X part and Z aprt are switched
steane_log_ops = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Z_L
                           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]) # X_L (2 x 14)

# Setting up the device and dataset
dataset = '/project/tbrun_769/qdec/datasets/[[7,1,3]]p0_1data20000.csv'
#dataset = '[[7,1,3]]p0_1data20000.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dl.dataloader(dataset, device)

# Defining the architecture
layersizes = [6, 20, 30, 45, 35, 14]
acts = [sigmoid, tanh, relu, tanh, sigmoid]
QuantumDecoderNet = Net(layersizes, acts).to(device)

# Filenames
if sys.argv[1] == "n":
  timestamp = str(datetime.datetime.now())[5:23].replace(":", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
elif sys.argv[1] == "e":
  timestamp = sys.argv[2]
mod_filename = "/project/tbrun_769/qdec/models/model_"+timestamp+".pt"
acc_filename = "/project/tbrun_769/qdec/models/acc_"+timestamp+".pkl"

# Hyperparameters
kwargs = {'epochs': 5,
          'learningRate': 10**-4,
          'momentum': 0.9,
          'num_random_trials': 30,
          'precision': 5,
          'criterion': nn.BCELoss(),
          'mod_filename': mod_filename,
          'acc_filename': acc_filename,
          'stabs': steane_stabs,
          'log_ops': steane_log_ops}

train(QuantumDecoderNet, *data[:4], **kwargs)




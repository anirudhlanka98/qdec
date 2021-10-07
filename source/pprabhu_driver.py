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

golay_stabs = np.array([[0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,1,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[1,0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]])
			
golay_log_ops = np.array([[0,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0]])


# Setting up the device and dataset
dataset = '/project/tbrun_769/qdec/datasets/[[23,1,7]]p0_075data150000.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dl.dataloader(dataset, device)

# Defining the architecture
#layersizes = [6, 20, 30, 35, 45, 14]
layersizes = [22,50,50,60,60,46]
acts = [tanh, tanh, tanh, tanh, sigmoid]
QuantumDecoderNet = Net(layersizes, acts, device)
num_epochs = 100
learning_rate = 0.05
learning_rate_final_epoch =0.001  # must be less than learning_rate
trials_at_end = 50
trials_offset = 10

# Filenames
if sys.argv[1] == "n":
  timestamp = str(datetime.datetime.now())[5:23].replace(":", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
  print(timestamp, layersizes, acts, num_epochs, learning_rate, trials_at_end)
elif sys.argv[1] == "e":
  timestamp = sys.argv[2]
mod_filename = "/project/tbrun_769/qdec/models/model_"+timestamp+".pt"
acc_filename = "/project/tbrun_769/qdec/models/acc_"+timestamp+".pkl"

# Hyperparameters
kwargs = {'epochs': num_epochs,
          'learningRate': learning_rate,
          'learningLast': learning_rate_final_epoch
          'momentum': 0.9,
          'num_random_trials': trials_at_end,
	  			'trials_offset':trials_offset,
          'precision': 5,
          'criterion': nn.BCELoss(),
          'mod_filename': mod_filename,
          'acc_filename': acc_filename,
          'stabs': golay_stabs,
          'log_ops': golay_log_ops}

train(QuantumDecoderNet, *data[:4], **kwargs)



import dataloader as dl
from neural_net import Net, train
import sys
import datetime
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Activation functions
elu = nn.ELU() # Exponential linear function
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
			[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1],
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


color_12_2_3_stabs = np.array([[1,1,0,1,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,1,0,1,1,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,1,1,1,1,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,0,1,1,0, 0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,1,0,1,1, 0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0, 1,1,0,1,1,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0, 0,1,1,0,1,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,1,1,1,1,1,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,1,1,0,1,1,0],
[0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1,1,0,1,1]
])

color_12_2_3_log_ops = np.array([[1,1,1,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0],
[1,0,0,1,0,0,1,0,0,1,0,0, 0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0, 1,0,0,1,0,0,1,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,1,1,1]
])

# Setting up the device and dataset

#dataset = '/project/tbrun_769/qdec/datasets/[[23,1,7]]Corrdata200000.csv'
dataset = '/project/tbrun_769/qdec/datasets/[[7,1,3]]Corrdata50000.csv'
#dataset = '/project/tbrun_769/qdec/datasets/[[12,2,3]]Corrdata75000.csv'
checkpoint_dir = None
num_samples=25
max_num_epochs=100
gpus_per_trial=0

# Defining the architecture
layersizes = [6, 14]
#layersizes = [22,70,100,100,46]
#layersizes = [10, 50, 100, 24]
acts = [elu, sigmoid]

num_epochs = max_num_epochs
#learning_rate = 0.0005
#learning_rate_final_epoch =0.0001 # must be less than learning_rate
#trials_at_end = 35
#trials_offset = 10

config = {
        "trials": tune.quniform(lower=2, upper=2, q=2),
        "lr": tune.loguniform(5e-6, 1e-4),
}

# Filenames
if sys.argv[1] == "n":
  timestamp = str(datetime.datetime.now())[5:23].replace(":", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
  print(timestamp, layersizes, acts, num_epochs)
elif sys.argv[1] == "e":
  timestamp = sys.argv[2]
mod_filename = "/project/tbrun_769/qdec/models/model_"+timestamp
acc_filename = "/project/tbrun_769/qdec/models/acc_"+timestamp

# Hyperparameters
kwargs = {'epochs': num_epochs,
         # 'learningRate': learning_rate,
         # 'learningLast': learning_rate_final_epoch,
          'momentum': 0.9,
         # 'num_random_trials': config["trials"],
	       # 'trials_offset':trials_offset,
          'precision': 5,
          'criterion': nn.BCELoss(),
          'mod_filename': mod_filename,
          'acc_filename': acc_filename,
          'stabs': steane_stabs,
          'log_ops': steane_log_ops,
          'layersizes': layersizes,
	  'acts': acts,
	  'dataset': dataset
}

# Ray tune wrappers
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=20,
    reduction_factor=1.5)
reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "x_log_val_epoch", "epoch"])
result = tune.run(
    partial(train, **kwargs),
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)


best_trial = result.get_best_trial("accuracy", "max", "last")
if best_trial is None:
  print("best_trial is None. Check output logs.")
else:
  print("Best trial config: {}".format(best_trial.config))
  print("Best trial final validation loss: {}".format(
      best_trial.last_result["loss"]))
  print("Best trial final validation accuracy: {}".format(
      best_trial.last_result["accuracy"]))

#train(QuantumDecoderNet, checkpoint_dir, device, *data[:4], **kwargs)


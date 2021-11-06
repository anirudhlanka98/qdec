import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import pickle as pkl
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import dataloader as dl

class Net(nn.Module):

  def __init__(self, layersizes, acts):
    super(Net, self).__init__()
    self.acts = acts
    self.syndrome_input = nn.Linear(in_features=layersizes[0], out_features=layersizes[1])
    self.hidden = [nn.Linear(in_features=layersizes[j + 1], out_features=layersizes[j + 2]) for j in range(len(acts) - 3)]
    self.error_dist = nn.Linear(in_features=layersizes[-2], out_features=layersizes[-1])

  def forward(self, syndrome):

    num_layers = len(self.acts)
    layers = [self.syndrome_input] + self.hidden + [self.error_dist]
    a_0 = self.acts[0](syndrome)

    def arch(input, l):
      z_l = layers[l](input)
      a_l = self.acts[l+1](z_l)
      if l < num_layers-2:
        return arch(a_l, l+1)
      else:
        return a_l
    
    return arch(a_0, 0)


def train(config, checkpoint_dir =None,  **kwargs):
  loss_arr = []
  train_acc_codespace, valid_acc_codespace = [], []
  train_acc_x, valid_acc_x = [], []
  train_acc_z, valid_acc_z = [], []
  QuantumDecoderNet = Net(kwargs['layersizes'], kwargs['acts'])
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
      net = nn.DataParallel(net)
  QuantumDecoderNet.to(device) 

  data = dl.dataloader(kwargs['dataset'],device)
  train_syndromes, train_error_labels, valid_syndromes, valid_error_labels = data[:4]
  print(len(train_syndromes), len(valid_error_labels))  
  optimizer = optim.Adam(QuantumDecoderNet.parameters(), lr = config['lr'], betas = (0.9, 0.99), eps = 1e-08, weight_decay = 10**-4, amsgrad = False)

  if checkpoint_dir:
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
    net.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  for epoch in range(kwargs['epochs']):
    for idx, syndrome in enumerate(train_syndromes):
      syndrome, train_error_labels[idx] = syndrome.to(device), train_error_labels[idx].to(device)
      optimizer.zero_grad() # Initializing the gradients to zero
      output = QuantumDecoderNet.forward(syndrome)
      loss = kwargs['criterion'](output, train_error_labels[idx])
      loss.backward()
      optimizer.step()
    
    # Measure training and validation accuracy for each epoch
    train_acc_codespace_epoch, train_acc_x_epoch, train_acc_z_epoch = accuracy(QuantumDecoderNet, config, train_syndromes, train_error_labels, **kwargs)
    valid_acc_codespace_epoch, valid_acc_x_epoch, valid_acc_z_epoch = accuracy(QuantumDecoderNet, config, valid_syndromes, valid_error_labels, **kwargs)
    train_acc_codespace.append(train_acc_codespace_epoch)
    train_acc_x.append(train_acc_x_epoch)
    train_acc_z.append(train_acc_z_epoch)
    valid_acc_codespace.append(valid_acc_codespace_epoch)
    valid_acc_x.append(valid_acc_x_epoch)
    valid_acc_z.append(valid_acc_z_epoch)
    loss_epoch = loss.cpu().detach().numpy()
    loss_arr.append(loss_epoch)
    print("Epoch {}: Loss = {}".format(epoch+1, round(float(loss_epoch), kwargs['precision'])), flush=True, end = ', ')
    print("Training (Code) = {}, Validation (Code) = {}".format(round(train_acc_codespace_epoch, kwargs['precision']), round(valid_acc_codespace_epoch, kwargs['precision'])), flush=True, end = ', ')
    print("Training (X) = {}, Validation (X) = {}".format(round(train_acc_x_epoch, kwargs['precision']), round(valid_acc_x_epoch, kwargs['precision'])), flush=True, end = ', ')
    print("Training (Z) = {}, Validation (Z) = {}".format(round(train_acc_z_epoch, kwargs['precision']), round(valid_acc_z_epoch, kwargs['precision'])), flush=True)
    torch.save(QuantumDecoderNet, kwargs['mod_filename']+"_"+str(round(config['lr'],6))+"_"+ str(int(config['trials']))+ ".pt")

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((QuantumDecoderNet.state_dict(), optimizer.state_dict()), path)
  #  print(train_acc_codespace_epoch)
    tune.report(loss=loss_epoch, accuracy=train_acc_codespace_epoch, x_log_val_epoch=train_acc_x_epoch, epoch=epoch)
    results = [loss_arr, train_acc_codespace, train_acc_x, train_acc_z, valid_acc_codespace, valid_acc_x, valid_acc_z]
    with open(kwargs['acc_filename']+"_"+str(round(config['lr'],6))+"_"+ str(int(config['trials']))+ ".pkl", "wb") as file:
      pkl.dump(results, file)


def accuracy(QuantumDecoderNet, config, ds_synds, ds_error_labels, **kwargs):

  num_success = 0
  num_log_z = 0
  num_log_x = 0
  l = len(ds_synds)
  
  with torch.no_grad():
    for idx in range(l):
      output = QuantumDecoderNet.forward(ds_synds[idx]).cpu().detach().numpy()
      if kwargs["random_sampling"] == True:
        succ, logx, logz = sample(output, idx, ds_synds, ds_error_labels, config, **kwargs)
      else:
        succ, logx, logz = sample_threshold(0.5, output, idx, ds_synds, ds_error_labels, **kwargs)	
      num_success += succ
      num_log_x += logx
      num_log_z += logz

  codespace_acc = num_success / l
  if num_success > 0:
    x_space_acc = 1 - (num_log_x / num_success)
    z_space_acc = 1 - (num_log_z / num_success)	
  else:
    x_space_acc = -1
    z_space_acc = -1
  return codespace_acc, x_space_acc, z_space_acc


def sample(output, idx, ds_synds, ds_error_labels, config, **kwargs):
  
  success, log_x, log_z = 0, 0, 0
  len_output = len(output)
  for _ in range(int(config["trials"])):
    a = np.random.uniform(size = (len_output, 1))
    b = [1 if output[i] > a[i] else 0 for i in range(len_output)]
    predicted_syndrome = np.dot(kwargs['stabs'], b) % 2
    actual_syndrome = ds_synds[idx].cpu().detach().numpy()
    if np.array_equal(predicted_syndrome, actual_syndrome):
      success += 1
      corrected = np.array([int(b[i]) ^ int(ds_error_labels[idx].cpu().detach().numpy()[i]) for i in range(len(b))])
      log_error_exists = np.dot(kwargs['log_ops'], corrected.T) % 2
      if log_error_exists[0] == 1:
        log_z += 1
      if log_error_exists[1] == 1:
        log_x += 1
      break
  return success, log_x, log_z


def sample_threshold(th_value, output, idx, ds_synds, ds_error_labels, **kwargs):

  success, log_x, log_z = 0, 0, 0
  b = [1 if output[i] > th_value else 0 for i in range(len(output))]
  predicted_syndrome = np.dot(kwargs['stabs'], b) % 2
  actual_syndrome = ds_synds[idx].cpu().detach().numpy()
  if np.array_equal(predicted_syndrome, actual_syndrome):
    success += 1
    corrected = np.array([int(b[i]) ^ int(ds_error_labels[idx].cpu().detach().numpy()[i]) for i in range(len(b))])
    log_error_exists = np.dot(kwargs['log_ops'], corrected.T) % 2
    if log_error_exists[0] == 1:
      log_z += 1
    if log_error_exists[1] == 1:
      log_x += 1
  return success, log_x, log_z

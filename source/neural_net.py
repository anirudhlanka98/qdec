import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import pickle as pkl


class Net(nn.Module):

  def __init__(self, layersizes, acts, device):
    super(Net, self).__init__()
    self.acts = acts
    self.syndrome_input = nn.Linear(in_features=layersizes[0], out_features=layersizes[1], device=device)
    self.hidden = [nn.Linear(in_features=layersizes[j + 1], out_features=layersizes[j + 2], device=device) for j in range(len(acts) - 2)]
    self.error_dist = nn.Linear(in_features=layersizes[-2], out_features=layersizes[-1], device=device)

  def forward(self, syndrome):

    num_layers = len(self.acts)
    layers = [self.syndrome_input] + [self.hidden[i] for i in range(len(self.acts)-2)] + [self.error_dist]
    def arch(input, l):
      z_l = layers[l](input)
      a_l = self.acts[l](z_l)
      if l < num_layers-1:
        return arch(a_l, l+1)
      else:
        return a_l
    
    return arch(syndrome, 0)


def train(QuantumDecoderNet, *args, **kwargs):

  loss_arr = []
  train_acc_codespace, valid_acc_codespace = [], []
  train_acc_x, valid_acc_x = [], []
  train_acc_z, valid_acc_z = [], []
  train_syndromes, train_error_labels, valid_syndromes, valid_error_labels = args
  optimizer = optim.Adam(QuantumDecoderNet.parameters(), lr = kwargs['learningRate'], betas = (0.9, 0.99), eps = 1e-08, weight_decay = 10**-4, amsgrad = False)

  for epoch in range(kwargs['epochs']):
    for idx, syndrome in enumerate(train_syndromes):
      optimizer.zero_grad() # Initializing the gradients to zero
      output = QuantumDecoderNet.forward(syndrome)
      loss = kwargs['criterion'](output, train_error_labels[idx])
      loss.backward()
      optimizer.step()
    
    # Measure training and validation accuracy for each epoch
    train_acc_codespace_epoch, train_acc_x_epoch, train_acc_z_epoch = accuracy(QuantumDecoderNet, train_syndromes, train_error_labels, **kwargs)
    valid_acc_codespace_epoch, valid_acc_x_epoch, valid_acc_z_epoch = accuracy(QuantumDecoderNet, valid_syndromes, valid_error_labels, **kwargs)
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
    torch.save(QuantumDecoderNet, kwargs['mod_filename'])

  results = [loss_arr, train_acc_codespace, train_acc_x, train_acc_z, valid_acc_codespace, valid_acc_x, valid_acc_z]
  with open(kwargs['acc_filename'], "wb") as file:
    pkl.dump(results, file)


def accuracy(QuantumDecoderNet, ds_synds, ds_error_labels, **kwargs):

  num_success = 0
  num_log_z = 0
  num_log_x = 0
  l = len(ds_synds)
  
  with torch.no_grad():
    for idx in range(l):
      output = QuantumDecoderNet.forward(ds_synds[idx]).cpu().detach().numpy()
      for _ in range(kwargs['num_random_trials']):
        a = np.random.uniform(size = (len(output), 1))
        b = [1 if output[i] > a[i] else 0 for i in range(len(output))]
        predicted_syndrome = np.dot(kwargs['stabs'], b) % 2
        actual_syndrome = ds_synds[idx].cpu().detach().numpy()
        if np.array_equal(predicted_syndrome, actual_syndrome):
          num_success += 1
          corrected = np.array([int(b[i]) ^ int(ds_error_labels[idx].cpu().detach().numpy()[i]) for i in range(len(b))])
          log_error_exists = np.dot(kwargs['log_ops'], corrected.T) % 2
          if log_error_exists[0] == 1:
            num_log_z += 1
          if log_error_exists[1] == 1:
            num_log_x += 1
          break
  codespace_acc = num_success / l
  if num_success>0:
    x_space_acc = 1 - (num_log_x / num_success)
    z_space_acc = 1 - (num_log_z / num_success)	
  else:
    x_space_acc = 1
    z_space_acc = 1
  return codespace_acc, x_space_acc, z_space_acc





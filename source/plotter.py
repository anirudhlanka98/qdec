import pickle as pkl
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

filename = sys.argv[1]
figname = "/project/tbrun_769/qdec/plots/res_"+filename.split('_', 1)[1].split('.')[0]+".pdf"
pp = PdfPages(figname)

with open(filename, "rb") as input_file:
    loss_arr, train_acc_codespace, train_acc_x, train_acc_z, valid_acc_codespace, valid_acc_x, valid_acc_z = pkl.load(input_file)

epochs = len(train_acc_codespace)

fig, ax = plt.subplots()
ax.plot(np.arange(1, epochs+1), loss_arr)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss')
pp.savefig()

fig, ax = plt.subplots()
ax.plot(np.arange(1, epochs+1), train_acc_codespace)
ax.plot(np.arange(1, epochs+1), valid_acc_codespace)
ax.legend(['Training', 'Validation'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Codespace accuracy')
pp.savefig()

fig, ax = plt.subplots()
ax.plot(np.arange(1, epochs+1), train_acc_x)
ax.plot(np.arange(1, epochs+1), valid_acc_x)
ax.legend(['Training', 'Validation'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Logical X space accuracy')
pp.savefig()

fig, ax = plt.subplots()
ax.plot(np.arange(1, epochs+1), train_acc_z)
ax.plot(np.arange(1, epochs+1), valid_acc_z)
ax.legend(['Training', 'Validation'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Logical Z space accuracy')
pp.savefig()

pp.close()

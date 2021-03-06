# Non-topological quantum decoder with neural networks

### How to run:
1. Driver: ```python3 *_driver.py <n/e> <timestamp> ```
  * If running for the first time, use only ```n```. This creates 2 files:
    * ```models/mod_*.pt``` is the model file that updates after every epoch
    * ```models/acc_*.pkl``` is the pickled file of code, X and Z space accuracies for every epoch. Use it to plot the accuracy-epoch graph using plotter.py
  * Else, use ```e``` along with the corresponding timestamp to continue training
 
2. Plotter (Located in source/): ```python3 plotter.py <path_to_accuracy_file.pkl>```

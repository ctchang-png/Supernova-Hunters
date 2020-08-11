import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from dataset_loader import *
from model_loader import *
from plotting_utils import *


#########################
## Hardware management ##
#########################
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs", len(gpus))

#https://github.com/tensorflow/tensorflow/issues/7072 (karthikeyan19)
#No clue what this does but it fixes CUBLAS errors
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

strategy = tf.distribute.MirroredStrategy()
my_scope = strategy.scope()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
##################
## RUN TRAINING ##
##################

def run_training(model, train_ds, valid_ds, epochs, batch_size):
  print('\nTraining . . .\n')

  #callbacks/cp
  checkpoint_dir = './checkpoints/'
  log_dir = './logs/'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir),
              tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                save_weights_only=True)]
  
  if tf.__version__ in {'2.2.0', '2.3.0'}:
    #In 2.2 validation_data cannot be a from_generator dataset
    #If validation_data is not a dataset, validation_batch_size must be specified
    valid_x, valid_y = dataset2array(valid_ds)
    model.fit(train_ds, validation_data=(valid_x, valid_y), validation_batch_size=batch_size,
              epochs=epochs, callbacks=callbacks, verbose=2)
  else:
    raise Exception("Version not implemented. Verify which datasets can be added to model.fit")

######################
## Training/Results ##
######################

def train(architecture, data_type, folds, epoch):
  batch_size = 128
  super_batch_size = batch_size * strategy.num_replicas_in_sync
  #for model in os.listdir("./saved_models/"):
  #  os.remove(os.path.join("./saved_models/", model))
  #Windows says no

  dataset, _, _ = get_ds_fn(data_type)(folds, architecture, super_batch_size)
  print('tf version', tf.__version__)
  print()
  print("Training Model: " + str(architecture))
  print("Data method: {} with {} training folds".format(data_type, folds))
  print("Batch Size: " + str(super_batch_size))
  fold = 1
  with my_scope:
    for x_train, y_train, x_test, y_test in dataset:
      #test denotes validation in this case
      print()
      print("Training fold: " + str(fold))
      train_ds, valid_ds = make_batches(x_train, y_train, x_test, y_test, super_batch_size)
      model = get_model(architecture)
      run_training(model, train_ds, valid_ds, epoch, super_batch_size) #no clue which batch size to use here
      suffix = str(fold)
      model.save("./saved_models/" + "Model_" + suffix)
      fold += 1
  check_test_set(architecture)
  return 0

def check_test_set(architecture, folder="./saved_models/"):
  _, test_ds, f_test = make_std_ds(None, architecture, 32)
  models = load_models(folder)
  display_metrics(models, test_ds)
  #magnitude_study(models, test_ds, f_test)
  #display_predictions(models, test_ds, num=5)

def magnitude_study(models, test_ds, f_test):
  x_test, y_test = dataset2array(test_ds)
  predictions, labels = get_ensembled_predictions(models, x_test, y_test)
  _, thresh, _, _ = one_percent_fpr(labels, predictions, 0.01)
  predictions = predictions > thresh
  real_mask = np.where(labels==1)
  _, mask = np.unique(f_test[real_mask], return_index=True)
  f_test = file_array_to_list(f_test[real_mask][mask])
  mag_dict = load_magnitude_data()
  mags = []
  for f in f_test:
    mags.append(mag_dict[f])
  bins = np.arange(14,23,1)   #???
  mid_points = []
  for i in np.arange(14,22,1):
    mid_points.append(np.mean([i,i+1]))

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twinx()

  f1s = np.zeros((len(bins)-1,))

  n, bins, _ = ax1.hist(mags, bins=bins, alpha=0)
  bin_allocations = np.digitize(mags, bins)
  for j in range(1,len(bins)-1):
    if n[j-1] == 0:
      continue
    f1s[j] += f1_score(labels[real_mask][mask][bin_allocations==j], predictions[real_mask][mask][bin_allocations==j])

  ax1.plot(mid_points, f1s, color='#7A8C95', label="f1 score")
  ax1.set_ylim(0,1)
  ax1.set_zorder(ax2.get_zorder()+1)
  ax1.patch.set_visible(False)
  plt.title("F1 score by magnitude")
  plt.xlabel("magnitude")
  plt.ylabel("f1 score")
  plt.legend()
  plt.show()

def compare_models():
  plots = dict()

  for ensemble in os.listdir("./Ensembles/"):
    if ensemble == "OLD":
      continue
    print(ensemble)
    name = ensemble.split("_")[0]
    method = ensemble.split("_")[1]
    _, test_ds, _ = make_std_ds(None, name, 32)
    x_test, y_test = dataset2array(test_ds)
    predictions, labels = get_ensembled_predictions(load_models("./Ensembles/" + ensemble), x_test, y_test)
    percent_fpr=0.01
    FoM, _, fpr, tpr = one_percent_fpr(labels, predictions, percent_fpr)
    mdr = 1-tpr
    print(ensemble + " " + str(FoM))
    if name not in plots:
      plots[name] = []
    plots[name].append((fpr, mdr, method, FoM))

  #Data method comparison
  for name in plots:
    plt.figure()
    plt.title(name + ": method comparison")
    plt.ylabel("False Positive Rate (FPR)")
    plt.xlabel("Missed Detection Rate (MDR)")
    plt.xlim((0,0.40))
    plt.ylim((0,0.30))
    plt.plot([0,1],[0.01,0.01], 'k--')
    for fpr, mdr, method, _ in plots[name]:
      plt.plot(fpr, mdr, label=method, color=name2color(method))
    plt.legend()
  plt.show()

  #Architecture comparison
  plt.figure()
  plt.title("Architecture comparison")
  plt.ylabel("False Positive Rate (FPR)")
  plt.xlabel("Missed Detection Rate (MDR)") 
  plt.xlim((0,0.40))
  plt.ylim((0,0.30))
  plt.plot([0,1],[0.01,0.01], 'k--')
  plt.plot([0,1],[0.05,0.05], 'k--')
  for ensemble in ["CustomResNet100x100_Fivefold_MDR11", "BaselineFlat_Fivefold_MDR30", \
                   "CustomResNet50x50_Fivebag_MDR14"]:
    name = ensemble.split("_")[0]
    method = ensemble.split("_")[1]
    for fpr, mdr, m, FoM in plots[name]:
      if method == m:
        plt.plot(fpr, mdr, label=name, color=name2color(name))
        break
    print(name + " " + str(FoM))
  plt.legend()
  plt.show()


  #f1 scores
  _, ax = plt.subplots(figsize=(9.2, 5))
  plt.title("f1 scores")
  ax.invert_yaxis()
  ax.xaxis.set_visible(False)
  ax.set_xlim((0,1))
  for i, ensemble in enumerate(["CustomResNet100x100_Fivefold_MDR11", \
                   "CustomResNet50x50_Fivebag_MDR14",
                   "BaselineFlat_Fivefold_MDR30"]):
    name = ensemble.split("_")[0]
    _, test_ds, _ = make_std_ds(None, name, 32)
    x_test, y_test = dataset2array(test_ds)
    predictions, labels = get_ensembled_predictions(load_models("./Ensembles/" + ensemble), x_test, y_test)
    percent_fpr=0.01
    FoM, threshold , fpr, tpr = one_percent_fpr(labels, predictions, percent_fpr)
    score = f1_score(labels, predictions>threshold)
    print(ensemble + " " + str(score))
    ax.barh(name, score, color=name2color(name))
    ax.text(score/2, i, "{:0.3f}".format(score), ha='center', va='center')
  plt.show()

#train(architecture, data_method, n_folds, epochs)
#train("CustomResNet100x100", "Bagging", 5, 75)
#check_test_set(architecture, folder="saved_models/")
#check_test_set("CustomResNet100x100", folder="Ensembles/CustomResNet100x100_Fivefold_MDR11")
#compare_models()

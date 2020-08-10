import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K

############################
## Predictions/Ensembling ##
############################

def get_ensembled_predictions(models, X, Y):
  predictions = np.zeros(np.shape(Y))
  for model in models:
    pred = model.predict(X)
    predictions += pred
  predictions /= len(models)
  labels = np.reshape(Y, (-1,))
  predictions = np.reshape(predictions, (-1,))
  return predictions, labels

########################
##  Plotting Helpers  ##
########################
def display_predictions(models, dataset, num=5):
  X, Y = dataset2array(dataset)

  predictions, labels = get_ensembled_predictions(models, X, Y)

  for i in range(num):
    plt.figure()
    img = np.reshape(X[i,...], (20,20))
    plt.imshow(img)
    plt.title("Image {:1.0f}   Label: {:1.0f}  Prediction: {:0.2f}".format(i+1, labels[i], predictions[i]))
    plt.show()

def plot_loss(logs):
  plt.figure()
  plt.plot(logs["x"], logs["training_loss"], 'r-', logs["x"], logs["testing_loss"], 'b-')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  red_line = mpatches.Patch(color='red', label='training')
  blue_line = mpatches.Patch(color='blue', label='testing')
  plt.legend(handles=[red_line, blue_line])
  plt.xlim(left=0)

def one_percent_fpr(y, pred, fom):
  try:
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1.0)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
  except IndexError:
    FoM = 1.0
    threshold = 1.0
    fpr=[1.0]
    tpr=[0.0]
  return FoM , threshold, fpr, tpr

def plot_fpr_mdr(fpr, mdr):
  plt.figure()
  plt.title("FPR v. MDR")
  plt.plot(mdr, fpr, 'b-')
  plt.xlim(left=0)
  plt.ylim(bottom=0)
  plt.xlabel('Missed Detection Rate (MDR)')
  plt.ylabel('False Positive Rate (FPR)')
  plt.plot([0,1],[0.01,0.01], 'k--')

def plot_prediction_distribution(results):
  plt.figure()
  plt.title('Prediction distribution')
  plt.plot(np.ones(np.shape(results)), results, 'rx')
  plt.xlim(left=0, right=2)
  plt.ylim(bottom=0, top=1)

def display_metrics(models, dataset):
  X, Y = dataset2array(dataset)
  predictions, labels = get_ensembled_predictions(models, X, Y)
  percent_fpr=0.01
  FoM, threshold, fpr, tpr = one_percent_fpr(labels, predictions, percent_fpr)
  print("FoM: {:0.3f}".format(FoM))
  print("Threshold: {:0.3f}".format(threshold))
  plot_fpr_mdr(fpr, 1-tpr)
  plot_prediction_distribution(predictions)
  plt.show()

def dataset2array(dataset):
  dataset = dataset.unbatch()
  arr = tfds.as_numpy(dataset)
  x_list = []
  y_list = []
  for ex in arr:
    x_list.append(ex[0])
    y_list.append(ex[1])
  X = np.stack(x_list)
  Y = np.concatenate(y_list)
  Y = np.reshape(Y, (-1,1))
  return X, Y


def name2color(s):
  D = {"Standard": "#ec5d37",
       "Fivefold": "#ffc800",
       "Fivebag": "#f64975",
       "CustomResNet100x100": "#00cfcc",
       "CustomResNet50x50": "#e898ac",
       "BaselineFlat": "#a9a9a9"}
  return D[s]
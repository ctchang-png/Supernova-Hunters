import scipy
from scipy import io
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, AveragePooling2D
from tensorflow.keras import Model
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt

#tf.config.experimental.list_physical_devices('GPU')

###################
##  IMPORT DATA  ##
###################

def load_data(model_architecture):
    out = dict()
    if model_architecture in ["CustomResNet50x50"]:
        data = scipy.io.loadmat("datasets/3pi_50x50_channels3_formatted")

        x_train = data['x_train'] #(m, 100,100,3)
        y_train = data['y_train'] #(m, 1)
        #print(np.shape(x_train))

        out["x_train"] = x_train
        out["y_train"] = y_train

        x_test = data['x_test'] #(m, 100,100,3)
        y_test = data['y_test']

        out["x_test"] = x_test
        out["y_test"] = y_test
        out["f_test"] = data["test_files"]
        out["f_train"] = data["train_files"]
    elif model_architecture in ["CustomResNet100x100"]:
        data = scipy.io.loadmat("datasets/3pi_100x100_channels3_formatted")

        x_train = data['x_train'] #(m, 100,100,3)
        y_train = data['y_train'] #(m, 1)
        #print(np.shape(x_train))

        out["x_train"] = x_train
        out["y_train"] = y_train

        x_test = data['x_test'] #(m, 100,100,3)
        y_test = data['y_test']

        out["x_test"] = x_test
        out["y_test"] = y_test
        out["f_test"] = data["test_files"]
        out["f_train"] = data["train_files"]
    elif model_architecture in ["BaselineFlat", "CustomResNet20x20"]:
        #data = scipy.io.loadmat("datasets/3pi_20x20_small")
        #data = scipy.io.loadmat("datasets/psst_20x20_large")
        data = scipy.io.loadmat("datasets/3pi_20x20_xpert_testset")
        #data = scipy.io.loadmat("datasets/3pi_20x20_xpert_halfreal")
        
        x_train = data['x_train'] #(m, 400)
        y_train = data['y_train'] #(m, 1)

        out["x_train"] = x_train
        out["y_train"] = y_train

        x_test = data['x_test']
        y_test = data['y_test']

        out["x_test"] = x_test
        out["y_test"] = y_test
        
        if "test_files" in data.keys():
            out["f_test"] = data["test_files"]
        else:
            out["f_test"] = None
        if "train_files" in data.keys():
            out["f_train"] = data["train_files"]
        else:
            out["f_train"] = None
    else:
        print("Dataset not yet implemented") #future datasets

    return out

################
##  DATASETS  ##
################

def make_std_ds(placeholder, model_architecture, batch_size):
  data = load_data(model_architecture)
  X = data["x_train"]
  Y = data["y_train"]
  x_test = data["x_test"]
  y_test = data["y_test"]
  #f_test = data["f_test"]
  #f_train = data["f_train"] <-------------get later
  m = np.shape(Y)[0]

  def gen():
    x_train, x_valid = X[:-(m//5)], X[-(m//5):]
    y_train, y_valid = Y[:-(m//5)], Y[-(m//5):]
    yield x_train,y_train,x_valid,y_valid
  
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
  return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64)), test_ds

def make_kfold_ds(n_splits, model_architecture, batch_size):
  data = load_data(model_architecture)
  X = data["x_train"]
  Y = data["y_train"]
  x_test = data["x_test"]
  y_test = data["y_test"]
  #f_test = data["f_test"]
  #f_train = data["f_train"] <-------------get later

  def gen():
    for train_index, test_index in KFold(n_splits=n_splits).split(X):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      yield X_train,y_train,X_test,y_test
  
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
  return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64)), test_ds

def make_bag(X, Y):
  m = np.shape(Y)[0]
  L_X = []
  L_Y = []
  # See if there's a way to vectorize this
  for _ in range(m):
    idx = np.random.randint(0, m)
    L_X.append(X[idx,...])
    L_Y.append(Y[idx,...])
  return np.stack(L_X), np.stack(L_Y)

def make_bagged_ds(n_bags, model_architecture, batch_size):
  data = load_data(model_architecture)
  X = data["x_train"]
  Y = data["y_train"]
  x_test = data["x_test"]
  y_test = data["y_test"]
  #f_test = data["f_test"]
  #f_train = data["f_train"] <-------------get later
  m = np.shape(Y)[0]
  def gen():
    for _ in range(n_bags):
      x_train, y_train = make_bag(X[:-(m//5),...], Y[:-(m//5),...])
      x_test, y_test = X[-(m//5):,...], Y[-(m//5):,...]
      yield x_train,y_train,x_test,y_test

  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
  return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64)), test_ds

def make_skfold_ds(n_splits, model_architecture, batch_size):
  data = load_data(model_architecture)
  X = data["x_train"]
  Y = data["y_train"]
  x_test = data["x_test"]
  y_test = data["y_test"]
  #f_test = data["f_test"]
  #f_train = data["f_train"] <-------------get later

  def gen():
    for train_index, test_index in StratifiedShuffleSplit(n_splits=n_splits, test_size=0.20).split(X, Y):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      yield X_train,y_train,X_test,y_test
  
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
  return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64)), test_ds

def make_batches(x_train, y_train, x_valid, y_valid, batch_size):
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True)

  valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size, drop_remainder=True)
  return train_ds, valid_ds

def get_ds_fn(data_type):
  if data_type == "Bagging":
    return make_bagged_ds
  elif data_type == "Kfold":
    return make_kfold_ds
  elif data_type == "Standard":
    return make_std_ds
  elif data_type == "Skfold":
    return make_skfold_ds
  else:
    raise NotImplementedError

#for 3pi_20x20_small
def load_magnitude_data():
  D = dict()
  for row in open('./datasets/test_real_3pi_c1_Stl.csv').readlines():
    key = row.split(',')[6].split('/')[-1]
    D[key] = float(row.split(',')[4])
  for row in open('./datasets/test_bogus_3pi_c1_Stl.csv').readlines():
    key = row.split(',')[6].split('/')[-1]
    D[key] = float(row.split(',')[4])
  return D

def file_array_to_list(file_array):
  return [f.strip() for f in file_array]
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
  MaxPool2D, AveragePooling2D, BatchNormalization, Add, Input, \
  GlobalAveragePooling2D, Dropout, Reshape
from tensorflow.keras import Model, regularizers
import tensorflow.keras.backend as K
from sklearn.metrics import roc_curve

#####################
## Network Blocks  ##
#####################

def res_block(input_data, filters, conv_size):
  x = Conv2D(filters, conv_size, activation='relu', padding='same', 
             kernel_regularizer='l2', bias_regularizer='l2')(input_data)
  x = BatchNormalization()(x)
  x = Conv2D(filters, conv_size, activation=None, padding='same',
             kernel_regularizer='l2', bias_regularizer='l2')(x)
  x = BatchNormalization()(x)
  x = Add()([x, input_data])
  x = tf.keras.layers.Activation('relu')(x)
  return x

######################
## Training Metrics ##
######################

#WIP
def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count
'''
def numpy_one_percent_fpr(y_true, y_pred):
  fom = 0.01
  try:
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
  except IndexError:
    FoM = 1.0
    threshold = 1.0
    fpr=[1.0]
    tpr=[0.0]
  return FoM , threshold, fpr, tpr

@tf.function
def tf_one_percent_fpr(y_true, y_pred):
  FoM, threshold, fpr, tpr = tf.numpy_function(numpy_one_percent_fpr, [y_true, y_pred], tf.float32)
  return FoM, threshold, fpr, tpr

def mdr_m(y_true, y_pred):
  FoM, _, _, _ = tf_one_percent_fpr(y_true, y_pred)
  return FoM
'''
# ^^^doesn't work b/c np.where tries to iterate over a tensor when autographed
'''
def mdr_m(y_true, y_pred):
  fom = 0.01
  thresholds = [0 + (1/1000)*i for i in range(1000)]
  try:
    fp = tf.compat.v1.metrics.false_positives_at_thresholds(y_true, y_pred, thresholds)
    tp = tf.compat.v1.metrics.true_positives_at_thresholds(y_true, y_pred, thresholds)
    p = tf_count(y_true, 1.0)
    n = tf_count(y_true, 0.0)
    fpr = fp/n
    tpr = tp/p
    mdr = 1-tpr
    FoM = tf.gather(mdr, tf.where(fpr<=fom))[0][-1] # MDR at 1% FPR
    threshold = tf.gather(mdr, tf.where(fpr<=fom))[0][-1]
  except IndexError:
    FoM = 1.0
    threshold = 1.0
    fpr=[1.0]
    tpr=[0.0]
  return FoM
'''
# ^^^doesn't work b/c "ValueError: tf.function-decorated function tried to create variables on non-first call"
# Error occurs on fp = tf.... line
# No clue how to fix that



#These assume a .50 threshold, add roc thresh later
#Update roc thresh is probably not feasible b/c weird graph exec things
def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

#soft f1 for loss (differentiable)
def f1_loss(y_true, y_pred):
  tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
  tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
  fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
  fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

  p = tp / (tp + fp + K.epsilon())
  r = tp / (tp + fn + K.epsilon())

  f1 = 2*p*r / (p+r+K.epsilon())
  f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
  return 1 - K.mean(f1)

###################
## Architectures ##
###################

def make_BaselineCNN():
  reg = 0.0001
  inputs = Input(shape=(50,50,3))
  x = Conv2D(16, 2, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(inputs)
  x = Conv2D(32, 2, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  x = Conv2D(64, 2, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  x = Conv2D(128, 2, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Dense(512, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  x = Dropout(0.5)(x)
  q = Dense(1, activation='sigmoid', name='q', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  model = Model(inputs, q)
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                metrics=[f1_m])
  return model


def make_BaselineFlat():
  reg=0.0001
  inputs = Input(shape=(400,))
  x = Dense(500, activation='relu',
    bias_regularizer=regularizers.l2(reg),
    kernel_regularizer=regularizers.l2(reg))(inputs)
  x = Dense(300, activation='relu',
    bias_regularizer=regularizers.l2(reg),
    kernel_regularizer=regularizers.l2(reg))(x)  
  x = Dense(10, activation='relu',
    bias_regularizer=regularizers.l2(reg),
    kernel_regularizer=regularizers.l2(reg))(x)
  q = Dense(1, activation='sigmoid',
    bias_regularizer=regularizers.l2(reg),
    kernel_regularizer=regularizers.l2(reg))(x)
  model = Model(inputs, q)
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                metrics=[f1_m])
  return model


def make_CustomResNet():
  inputs = tf.keras.Input(shape=(100, 100, 3))
  x = Conv2D(16, 3, padding='same',
    kernel_regularizer='l2', bias_regularizer='l2')(inputs)
  for _ in range(1):
    x = res_block(x, 16, 3)
  x = Conv2D(32, 3, activation='relu', padding='same',  
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  for _ in range(1):
    x = res_block(x, 32, 3)
  x = Conv2D(64, 3, activation='relu', padding='same',  
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  for _ in range(1):
    x = res_block(x, 64, 3)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Dense(512, activation='relu', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation='relu', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  x = Dropout(0.5)(x)
  outputs = Dense(1, activation='sigmoid', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  model = Model(inputs, outputs)
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
                metrics=[f1_m]) #look into mdr
  return model

def make_LargeResNet():
  inputs = tf.keras.Input(shape=(100, 100, 3))
  x = Conv2D(16, 3, padding='same',
    kernel_regularizer='l2', bias_regularizer='l2')(inputs)
  for _ in range(4):
    x = res_block(x, 16, 3)
  x = Conv2D(32, 3, activation='relu', padding='same',  
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  for _ in range(4):
    x = res_block(x, 32, 3)
  x = Conv2D(64, 3, activation='relu', padding='same',  
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  for _ in range(4):
    x = res_block(x, 64, 3)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Dense(512, activation='relu', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  x = Dropout(0.5)(x)
  x = Dense(256, activation='relu', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  x = Dropout(0.5)(x)
  outputs = Dense(1, activation='sigmoid', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  model = Model(inputs, outputs)
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
                metrics=[f1_m]) #look into mdr
  return model

def make_CustomResNet20x20():
  inputs = Input(shape=(400,))
  x = Reshape((20,20,1), input_shape=(400,))(inputs)
  x = Conv2D(16, 3, padding='same',
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  for _ in range(1):
    x = res_block(x, 16, 3)
  x = Conv2D(32, 3, activation='relu', padding='same',  
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  for _ in range(1):
    x = res_block(x, 32, 3)
  x = Conv2D(64, 3, activation='relu', padding='same',  
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  for _ in range(1):
    x = res_block(x, 64, 3)
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Dense(512, activation='relu', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  outputs = Dense(1, activation='sigmoid', 
    kernel_regularizer='l2', bias_regularizer='l2')(x)
  model = Model(inputs, outputs)
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                metrics=[f1_m])
  return model


def get_model(model_architecture):
  if model_architecture == "CustomResNet":
    model = make_CustomResNet()
  elif model_architecture == "BaselineCNN":
    model = make_BaselineCNN()
  elif model_architecture == "BaselineFlat":
    model = make_BaselineFlat()
  elif model_architecture == "CustomResNet20x20":
    model = make_CustomResNet20x20()
  elif model_architecture == "LargeResNet":
    model = make_LargeResNet()
  else:
    raise NotImplementedError("Model not yet implemented")
  #model.run_eagerly = False #Eagerly runs slower (especially on mirrored strategy) but cannot do custom metric/loss
                            #that doesn't use builtin tf/K functions
  return model

def load_models(directory):
  models = []
  for filename in os.listdir(directory):
    models.append(tf.keras.models.load_model(os.path.join(directory, filename), 
                  custom_objects={'f1_m':f1_m, 'f1_loss':f1_loss}))
  return models
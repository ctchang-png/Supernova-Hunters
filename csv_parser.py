import pickle

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, f1_score, accuracy_score

from keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras import regularizers

from keras.utils import np_utils

def one_percent_fpr(y, pred, fom):
  try:
    fpr, tpr, thresholds = roc_curve(y, pred)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
  except IndexError:
    FoM = 1.0
    threshold = 1.0
    fpr=[1.0]
    trp=[0.0]
  return FoM , threshold, fpr, tpr

def expert_data_getter(dummy):
  data = sio.loadmat('../data/3pi_20x20_skew2_signPreserveNorm.mat')
  x_train = data['X']
  y_train = np.squeeze(data['y'])
  f_train = np.squeeze(data['train_files'])
  
  x_test  = data['testX']
  y_test  = np.squeeze(data['testy'])
  f_test  = np.squeeze(data['test_files'])

  return x_train, y_train, f_train, x_test, y_test, f_test

def expert_cnn_data_getter(dummy):
  data = sio.loadmat('../data/3pi_20x20_skew2_signPreserveNorm.mat')
  x_train = np.reshape(data['X'], (data['X'].shape[0],20,20))[:,:,:,np.newaxis]
  y_train = np.squeeze(data['y'])
  f_train = np.squeeze(data['train_files'])
  
  x_test  = np.reshape(data['testX'], (data['testX'].shape[0],20,20))[:,:,:,np.newaxis]
  y_test  = np.squeeze(data['testy'])
  f_test  = np.squeeze(data['test_files'])

  return x_train, y_train, f_train, x_test, y_test, f_test

def volunteer_cnn_data_getter(user_name):

  data = sio.loadmat('../data/%s_labelled_subjects.mat'%(user_name))
  x_train = data['X']
  x_train = np.reshape(data['X'], (data['X'].shape[0],20,20))[:,:,:,np.newaxis]
  y_train = np.squeeze(data['y'])
  f_train = np.squeeze(data['train_files'])
  
  x_test  = data['testX']
  x_test  = np.reshape(data['testX'], (data['testX'].shape[0],20,20))[:,:,:,np.newaxis]
  y_test  = np.squeeze(data['testy'])
  f_test  = np.squeeze(data['test_files'])

  return x_train, y_train, f_train, x_test, y_test, f_test

def volunteer_data_expert_labels_getter(user_name):

  _, y_train_expert, f_train_expert_array, _, y_test_expert, f_test_expert_array = expert_data_getter('')
  f_train_expert = [f.strip() for f in f_train_expert_array]
  f_test_expert  = [f.strip() for f in f_test_expert_array]
  data = sio.loadmat('../data/%s_labelled_subjects.mat'%(user_name))
  x_train = data['X']
  f_train  = [f.strip() for f in data['train_files']]
  y_train = []
  for f in f_train:
    if f in set(f_train_expert):
      index = f_train_expert.index(f)
      y_train.append(y_train_expert[index])
  y_train = np.array(y_train)
  
  x_test  = data['testX']
  f_test  = [f.strip() for f in data['test_files']]
  y_test = []
  for f in f_test:
    if f in set(f_test_expert):
      index = f_test_expert.index(f)
      y_test.append(y_test_expert[index])
  y_test = np.array(y_test)
  assert x_test.shape[0] == y_test.shape[0]
  return x_train, y_train, f_train, x_test, y_test, f_test

def aggregated_volunteers_cnn_data_getter(dummy):

  classifications = pickle.load(open('../data/extracted_diff_classifications_3pi.pkl', 'rb'))
  
  x_train, _, f_train, x_test, _, f_test = expert_cnn_data_getter(dummy)
  
  train_files = [f.strip() for f in f_train]
  test_files  = [f.strip() for f in f_test]
  
  y_train = []
  for diff in train_files:
    annotations = classifications[diff]['annotations']
    y_train.append(np.mean(annotations)>0.5)
  y_train = np.array(y_train)
  
  y_test = []
  for diff in test_files:
    annotations = classifications[diff]['annotations']
    y_test.append(np.mean(annotations)>0.5)
  y_test = np.array(y_test)

  return x_train, y_train, f_train, x_test, y_test, f_test

def baseline_cnn_model_getter(input_shape, n_classes, lr=0.1, reg=0.0001, weights_file=None):

  # build the model for pre-training
  inputs = Input(shape=input_shape)
  x = Conv2D(16, 2, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(inputs)
  x = Conv2D(32, 2, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  x = Conv2D(64, 2, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  x = Flatten()(x)
  x = Dense(512, activation='relu', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)
  q = Dense(2, activation='softmax', name='q', \
    bias_regularizer=regularizers.l2(reg), \
    kernel_regularizer=regularizers.l2(reg))(x)

  model = Model(inputs, q)

  optimizer = SGD(lr=lr)
  model.compile(loss='categorical_crossentropy', \
                optimizer=optimizer, \
                metrics=['acc'])

  if weights_file:
    model.load_weights(weights_file)

  return model, 'baseline_cnn_model'

def file_array_to_list(file_array):
  return [f.strip() for f in file_array]

def load_magnitude_data():
  dict = {}
  for row in open('../data/test_real_3pi_c1_Stl.csv').readlines():
    key = row.split(',')[6].split('/')[-1]
    dict[key] = float(row.split(',')[4])
  for row in open('../data/test_bogus_3pi_c1_Stl.csv').readlines():
    key = row.split(',')[6].split('/')[-1]
    dict[key] = float(row.split(',')[4])
  return dict

def main():
  percent_fpr = 0.01
  volunteers = ['dennispattensr', 'BLGoodwin', 'Jose_Campos', 'kc5lei', 'SirUrielPerpetua', \
                'keeps2013', 'Wolffen99', 'nilium','aioeu', 'chrostek']

  bins = np.arange(14,23,1)
  mid_points = []
  for i in np.arange(14,22,1):
    mid_points.append(np.mean([i,i+1]))
  print(bins)
  print(mid_points)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twinx()

  ### Machine Learning Metrics And Plotting ###
  _, _, _, x_test, y_test, f_test = expert_cnn_data_getter('')

  real_mask = np.where(y_test==1)
  _, mask = np.unique(f_test[real_mask], return_index=True)
  f_test = file_array_to_list(f_test[real_mask][mask])

  mag_dict = load_magnitude_data()
  all_mags = [mag_dict[k] for k in f_test]
  print(np.min(all_mags), np.max(all_mags))

  model, model_name = baseline_cnn_model_getter((20,20,1), 2)

  preds_m = np.zeros((y_test[real_mask][mask].shape[0], 5*len(volunteers)))
  for trial in range(1,6):
    for j,volunteer in enumerate(volunteers):
      weights_file = '../results/%s/%s_%s_weights_trial_%d.h5'% \
        (model_name, volunteer, model_name, trial)
      model.load_weights(weights_file)
      print((j + (trial-1)*len(volunteers)))
      preds_m[:,(j+(trial-1)*len(volunteers))] += model.predict(x_test[real_mask][mask])[:,1]

  f1s_m = np.zeros((len(bins)-1,))

  n, bins, _ = ax2.hist(all_mags, bins=bins, alpha=.2, color='#F4B266')
  bin_allocations = np.digitize(all_mags, bins)
  print(bin_allocations)

  for j in range(1,len(bins)-1):
    if n[j-1] == 0:
      continue
    print(y_test[real_mask][mask][bin_allocations==j].shape, np.mean(preds_m[bin_allocations==j].shape))
    f1s_m[j] += f1_score(y_test[real_mask][mask][bin_allocations==j], np.mean(preds_m[bin_allocations==j], axis=1)>0.5)
    #f1s_m[j] += one_percent_fpr(y_test[real_mask][mask][bin_allocations==j], np.mean(preds_m[bin_allocations==j], axis=1), percent_fpr)[0]
  print(f1s_m)

  ### Expert Machine Learning Metrics And Plotting ###

  preds_em = np.zeros((y_test[real_mask][mask].shape[0], 5))
  for trial in range(1,6):
    weights_file = '../results/%s/%s_%s_weights_trial_%d.h5'% \
      (model_name, 'expert', model_name, trial)
    model.load_weights(weights_file)
    preds_em[:,trial-1] += model.predict(x_test[real_mask][mask])[:,1]
    print(one_percent_fpr(y_test[real_mask][mask], preds_em[:,trial-1], percent_fpr))

  f1s_em = np.zeros((len(bins)-1,))

  print(one_percent_fpr(y_test[real_mask][mask], np.mean(preds_em, axis=1), percent_fpr))
  for j in range(1,len(bins)-1):
    if n[j-1] == 0:
      continue
    f1s_em[j] += f1_score(y_test[real_mask][mask][bin_allocations==j], np.mean(preds_em[bin_allocations==j], axis=1)>0.5)
    #f1s_em[j] += one_percent_fpr(y_test[real_mask][mask][bin_allocations==j], np.mean(preds_em[bin_allocations==j], axis=1), percent_fpr)[0]
  print(f1s_em)

  ### Aggregated Machine Learning Metrics And Plotting ###
  preds_av = np.zeros((y_test[real_mask][mask].shape[0], 5))
  for trial in range(1,6):
    weights_file = '../results/%s/%s_%s_weights_trial_%d.h5'% \
      (model_name, 'aggregated_volunteer', model_name, trial)
    model.load_weights(weights_file)
    preds_av[:,trial-1] += model.predict(x_test[real_mask][mask])[:,1]
    print(one_percent_fpr(y_test[real_mask][mask], preds_av[:,trial-1], percent_fpr))

  f1s_av = np.zeros((len(bins)-1,))

  print(one_percent_fpr(y_test[real_mask][mask], np.mean(preds_av, axis=1), percent_fpr))
  for j in range(1,len(bins)-1):
    if n[j-1] == 0:
      continue
    f1s_av[j] += f1_score(y_test[real_mask][mask][bin_allocations==j], np.mean(preds_av[bin_allocations==j], axis=1)>0.5)
    #f1s_av[j] += one_percent_fpr(y_test[real_mask][mask][bin_allocations==j], np.mean(preds_av[bin_allocations==j], axis=1), percent_fpr)[0]
  print(f1s_av)

  ### Volunteer Metrics And Plotting ###
  """
  f1s_v = np.zeros((len(volunteers), len(bins)-1))
  for i,volunteer in enumerate(volunteers):
    _, _, _, x_test, y_test_v, f_test_v = volunteer_cnn_data_getter(volunteer)
    _, _, _, _, y_test_e, f_test_e = volunteer_data_expert_labels_getter(volunteer)
    real_mask = np.where(y_test_e == 1)
    _, mask = np.unique(f_test_v[real_mask], return_index=True)
    f_test_v = file_array_to_list(f_test_v[real_mask][mask])
    mags = []
    for f in f_test_v:
      mags.append(mag_dict[f])
    n, bins, _ = ax2.hist(mags, bins=bins, alpha=0)
    bin_allocations = np.digitize(mags, bins)
    for j in range(1,len(bins)):
      if n[j-1] == 0:
        continue
      f1s_v[i,j-1] += f1_score(y_test_e[real_mask][mask][bin_allocations==j], y_test_v[real_mask][mask][bin_allocations==j])
    print(volunteer, f1s_v[i,:])
  """
  _, _, _, x_test_a, y_test_a, f_test_a = aggregated_volunteers_cnn_data_getter('')
  _, mask = np.unique(f_test_a[real_mask], return_index=True)
  f_test_a = file_array_to_list(f_test_a[real_mask][mask])
  mags = []
  for f in f_test_a:
    mags.append(mag_dict[f])

  f1s_a = np.zeros((len(bins)-1,))

  n, bins, _ = ax2.hist(mags, bins=bins, alpha=0)
  bin_allocations = np.digitize(mags, bins)
  for j in range(1,len(bins)-1):
    if n[j-1] == 0:
      continue
    f1s_a[j] += f1_score(y_test[real_mask][mask][bin_allocations==j], y_test_a[real_mask][mask][bin_allocations==j])
    #f1s_a[j] += one_percent_fpr(y_test[real_mask][mask][bin_allocations==j], y_test_a[real_mask][mask][bin_allocations==j], percent_fpr)[0]
  print(f1s_a)

  ax1.plot(mid_points, f1s_a, color='#7A8C95', label='agg volunteers')
  ax1.plot(mid_points, f1s_m, color='#304D6D', label='individuals CNN')
  ax1.plot(mid_points, f1s_em, color='#E0777D', label='expert CNN')
  ax1.plot(mid_points, f1s_av, color='#8EA4D2', label='agg volunteers CNN')
  #ax1.fill_between(mid_points, np.mean(f1s_v, axis=0) - np.std(f1s_v, axis=0), np.mean(f1s_v, axis=0) + np.std(f1s_v, axis=0), color='#7A8C95', alpha=0.5)
  ax1.set_zorder(ax2.get_zorder()+1)
  ax1.set_ylim(0,1)
  #ax1.set_xlim(14,22)
  ax1.patch.set_visible(False)
  plt.legend()
  plt.show()



if __name__ == '__main__':
  main()
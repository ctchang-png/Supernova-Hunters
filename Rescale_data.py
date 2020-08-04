import scipy
from scipy import io
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, AveragePooling2D
from tensorflow.keras import Model
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os

'''
data100 = scipy.io.loadmat("datasets/Originals/3pi_100x100_channels3_skew2_signPreserveNorm")
#keys: ['__header__', '__version__', '__globals__', 'images', 'X', 'test_files']
data20 = scipy.io.loadmat("datasets/Originals/3pi_20x20_skew2_signPreserveNorm")
#keys: ['__header__', '__version__', '__globals__', 'testy', 'testX', 'train_files', 'y', 'X', 'test_files']
print(data100.keys())
print(data20.keys())

X = data100['X']
X = np.reshape(X, (-1, 3, 100, 100))
X = np.swapaxes(X, 1, 3)

newX = np.zeros((np.shape(X)[0], 50, 50, 3))

for i in range(np.shape(X)[0]):
    newX[i,...,0] = np.array(PIL.Image.fromarray(X[i,..., 0]).resize((50,50)))
    newX[i,...,1] = np.array(PIL.Image.fromarray(X[i,..., 1]).resize((50,50)))
    newX[i,...,2] = np.array(PIL.Image.fromarray(X[i,..., 2]).resize((50,50)))
X = newX

Y = np.concatenate((data20['y'], data20['testy'])) #Assume test is taken from the back
idx = np.shape(data20["y"])[0]
print(np.shape(X))
print(np.shape(Y))
x_test = X[idx:, ...]
x_train = X[:idx, ...]
y_test = Y[idx:, ...]
y_train = Y[:idx, ...]
'''

'''
predictions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(10):
    for j in range(3):
        plt.figure()
        img = x_test[i, ..., j]
        label = int(y_test[i, ...])
        plt.imshow(img)
        plt.title("Image " + str(i+1) + " Label: " + str(label) + " Prediction: " + str(j))
    plt.show()
'''


#m = 6916



X_list = []
Y_list = []
directory = "./datasets/psst_20x20_signPreserveNorm_2015-01-01_to_2015-04-01"
for filename in os.listdir(directory):
    if filename not in ["bogus_psst_20x20_signPreserveNorm_2015-01-01_to_2015-04-01_part26.mat",
                        "bogus_psst_20x20_signPreserveNorm_2015-01-01_to_2015-04-01_part27.mat",
                        "bogus_psst_20x20_signPreserveNorm_2015-01-01_to_2015-04-01_part28.mat",
                        "real_psst_20x20_signPreserveNorm_2015-01-01_to_2015-04-01_part1.mat",
                        "real_psst_20x20_signPreserveNorm_2015-01-01_to_2015-04-01_part2.mat",
                        "real_psst_20x20_signPreserveNorm_2015-01-01_to_2015-04-01_part3.mat"]:
        continue
    D = scipy.io.loadmat(os.path.join(directory, filename))
    #{"__header__", "__version__", "__globals__", "y", "X", "images"}
    print(D.keys())
    X_list.append(D["X"])
    y = np.reshape(D["y"], (-1, 1))
    Y_list.append(y)
X_tmp = np.concatenate(X_list, axis=0)
Y_tmp = np.concatenate(Y_list, axis=0)
m = np.shape(Y_tmp)[0]
perm = np.random.permutation(m)
X = np.zeros(np.shape(X_tmp))
Y = np.zeros(np.shape(Y_tmp))
for i, p in np.ndenumerate(perm):
    X[i,...] = X_tmp[p,...]
    Y[i,...] = Y_tmp[p,...]
print(np.shape(X))
print(np.shape(Y))


x_train = X
y_train = Y





D = scipy.io.loadmat("datasets/Originals/3pi_20x20_skew2_signPreserveNorm")
print(D.keys())
y_test = np.concatenate((D["y"], D["testy"]))
x_test = np.concatenate((D["X"], D["testX"]))

f_test = np.concatenate((D["train_files"], D["test_files"]))


data = {"x_test": x_test,
        "y_test": y_test,
        "x_train": x_train,
        "y_train": y_train,
        #"train_files": None,
        "test_files": f_test}
scipy.io.savemat("datasets/3pi_20x20_xpert_halfreal.mat", data, appendmat=False)
#keys: ['x_train', 'y_train', 'x_test', 'y_test']
#X is formatted to (None, 100, 100, 3)

################################################################################################################
# This program uses Keras and TensorFlow to implement a binary-classification DNN having 7 Features and 120
# weights. It also demonstrates how to input data from a csv file.
# Author: R. Bourquard - Dec 2020
# Modified by: D. Layton - Mar 2021
################################################################################################################

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# READ THE INPUT DATA
input_filename = 'Electrons.csv'
# TODO: Load data manually.
input_data = np.loadtxt(input_filename, dtype='float32', delimiter=",", skiprows=1)
print('input_data:', input_data.shape)
for i in range(0, input_data.shape[0]):
    if (input_data[i, 7] == -1): input_data[i,7] = 0
    if (input_data[i, 15] == -1): input_data[i, 15] = 0
# PRINT SOME OF THE INPUT DATA EXAMPLES
print('E1,\tpx1,\tpy1,\tpz1,\tpt1,\teta1,\tphi1,\tQ1,\tE2,\tpx2,\tpy2,\tpz2,\tpt2,\teta2,\tphi2,\tQ2,\tM')
for i in range(0, 19):
    print(round(input_data[i, 0], 2), "\t",
        round(input_data[i, 1], 2), "\t",
        round(input_data[i, 2], 2), "\t",
        round(input_data[i, 3], 2), "\t",
        round(input_data[i, 4], 2), "\t",
        round(input_data[i, 5], 2), "\t",
        round(input_data[i, 6], 2), "\t",
        round(input_data[i, 7], 2), "\t",
        round(input_data[i, 8], 2), "\t",
        round(input_data[i, 9], 2), "\t",
        round(input_data[i, 10], 2), "\t",
        round(input_data[i, 11], 2), "\t",
        round(input_data[i, 12], 2), "\t",
        round(input_data[i, 13], 2), "\t",
        round(input_data[i, 14], 2), "\t",
        round(input_data[i, 15], 2), "\t",
        round(input_data[i, 16], 2), "\t",
        sep=''
    )
# SPLIT THE INPUT DATA INTO A TRAINING DATASET AND A TESTING DATASET
training_split = 0.80
seed = 42
train_data, test_data = train_test_split(input_data[0:2000], train_size=training_split, random_state=seed)
print('train_data:', train_data.shape)
print('test_data:', test_data.shape)
print()

# TRAIN/TRUTH VALUES SPLIT
nFeatures = 16
ground_truth_col = 7
# for the training data
train_X = np.append(train_data[:,0:7], train_data[:, 8:], axis=1)   # The 7 features are in csv columns 2-8 [1:7]
train_truth = train_data[:,ground_truth_col]   # The 'survived' flag is in csv column 9 [8]
print('train_truth shape[0]', train_truth.shape[0])
# for the test data
test_X = np.append(test_data[:,0:7], test_data[:, 8:], axis=1)   # The 7 features are in csv columns 2-8 [1:7]
test_truth = test_data[:,ground_truth_col]   # The 'survived' flag is in column 9 [8]
print('train_X', train_X.shape)
print('train_truth', train_truth.shape)
print('test_X', test_X.shape)
print('test_truth', test_truth.shape)
print()



# NORMALIZE COLUMN DATA
scaler_obj = preprocessing.StandardScaler().fit(train_X)   # scaler_obj will scale each Feature (column) independently
train_X_scaled = scaler_obj.transform(train_X)  # scale each Training Feature (column)
test_X_scaled = scaler_obj.transform(test_X)  # scale each Test Feature (column)


print('train_X', train_X_scaled.shape)
print('train_truth', train_truth.shape)
print('test_X', test_X_scaled.shape)
print('test_truth', test_truth.shape)

# BUILD THE TENSORFLOW MODEL (This one has 1351 weights.)
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[nFeatures,]))
model.add(keras.layers.Dense(nFeatures + 1, activation='relu'))
model.add(keras.layers.Dense(nFeatures + 2, activation='relu'))
model.add(keras.layers.Dense(nFeatures + 2, activation='relu'))
model.add(keras.layers.Dense(nFeatures + 1, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=["accuracy"])

# FIND BEST VALUES FOR THE WEIGHTS
history = model.fit(train_X_scaled, train_truth, batch_size=1, epochs=20, validation_data=(test_X_scaled, test_truth))

# PRINT AND PLOT STATISTICS
score,accuracy = model.evaluate(train_X_scaled, train_truth, batch_size=16, verbose=0)
print("Train score (cost)       = {:.2f}".format(score))
print("Train accuracy (accuracy)= {:.2f}".format(accuracy))
score,accuracy = model.evaluate(test_X_scaled, test_truth, batch_size=16, verbose=0)
print("Test score (val_cost)    = {:.2f}".format(score))
print("Test accuracy (val_accuracy)= {:.2f}".format(accuracy))

# PLOT COST AND ACCURACY
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# PRINT A CONFUSION MATRIX FOR THE TEST EXAMPLES
probabilities = model.predict(test_X_scaled)   # the probability that the passengers survived
# convert the probabilities to predictions (0 or 1) for comparison with the ground truth
min_for_true = 0.5   # if the probability is >= 0.5, then assume the passenger survived
vector_int = np.vectorize(np.int32)
predictions = vector_int(probabilities + min_for_true)    # The int truncates any fractional part of the sum
((n_true_negatives, n_false_negatives), (n_false_positives, n_true_positives)) \
    = confusion_matrix(test_truth, predictions)
print()
print()
print('Confusion Matrix for Charge on Test Examples')
print()
print('     Negative Correctly Predicted:   ', n_true_negatives, '  ',
    n_false_negatives, ' :Negative Incorrectly Predicted')
print('     Positive Incorrectly Predicted:  ', n_false_positives, '  ',
    n_true_positives, ' :Positive Correctly Predicted')
print()

# PRINT SOME OF THE INPUT DATA EXAMPLES
print()
print()
print('E1,\tpx1,\t\tpy1,\tpz1,\tpt1,\teta1,\tphi1,\tQ1,\tE2,\tpx2,\tpy2,\tpz2,\tpt2,\teta2,\tphi2,\tQ2,\tM')
for i in range(0, 19):
    print(round(test_X[i, 0], 2), "\t",
        round(test_X[i, 1], 2), "\t",
        round(test_X[i, 2], 2), "\t",
        round(test_X[i, 3], 2), "\t",
        round(test_X[i, 4], 2), "\t",
        round(test_X[i, 5], 2), "\t",
        round(test_X[i, 6], 2), "\t",
        round(test_X[i, 7], 2), "\t",
        round(test_X[i, 8], 2), "\t",
        round(test_X[i, 9], 2), "\t",
        round(test_X[i, 10], 2), "\t",
        round(test_X[i, 11], 2), "\t",
        round(test_X[i, 12], 2), "\t",
        round(test_X[i, 13], 2), "\t",
        round(test_X[i, 14], 2), "\t",
        round(test_X[i, 15], 2), "\t",
        predictions[i]==1, "\t",
        test_truth[i]==1, "\t",
        sep=''
    )
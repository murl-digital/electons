################################################################################################################
# This program uses Keras and TensorFlow to implement a binary-classification DNN having 7 Features and 120
# weights. It also demonstrates how to input data from a csv file.
# Author: R. Bourquard - Dec 2020
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
# PRINT SOME OF THE INPUT DATA EXAMPLES
print('E1,\tpx1,\t\tpy1,\tpz1,\tpt1,\teta1,\tphi1,\tQ1,\tE2,\tpx2,\tpy2,\tpz2,\tpt2,\teta2,\tphi2,\tQ2,\tM')
for i in range(0, 19):
    print('{}\t'.format(input_data[i,0]),
        '{}\t'.format(input_data[i,1]),
        '{}\t'.format(input_data[i,2]),
        '{}\t'.format(input_data[i,3]),
        '{}\t'.format(input_data[i, 4]),
        '{}\t'.format(input_data[i, 5]),
        '{}\t'.format(input_data[i, 6]),
        '{}\t'.format(input_data[i, 7]),
        '{}\t'.format(input_data[i, 8]),
        '{}\t'.format(input_data[i, 9]),
        '{}\t'.format(input_data[i, 10]),
        '{}\t'.format(input_data[i, 11]),
        '{}\t'.format(input_data[i, 12]),
        '{}\t'.format(input_data[i, 13]),
        '{}\t'.format(input_data[i, 14]),
        '{}\t'.format(input_data[i, 15]),
        '{}\t'.format(input_data[i, 16]),
    )
# SPLIT THE INPUT DATA INTO A TRAINING DATASET AND A TESTING DATASET
training_split = 0.80
seed = 42
train_data, test_data = train_test_split(input_data, train_size=training_split, random_state=seed)
print('train_data:', train_data.shape)
print('test_data:', test_data.shape)
print()

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION #3:  Split-off the Features and the Ground Truth from the Training and Testing datafiles
# -----------------------------------------------------------------------------------------------------------------
# There are 8 features, in csv columns 1:8 [python columns 0:7].  However, column 1 [0] is the Passenger ID, which
# is (by my intuition) of no value to deciding who survived, so it is skipped.  Column 9 [8] is the 'survived'
# flag, which is our Ground Truth.  This code simply splits the Examples into a 2-D Feature matrix (rows=Examples,
# columns=the 7 Features), and the Ground Truth into a matching 1-D array.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# SEPARATE-OUT THE FEATURES AND THE GROUND TRUTH
nFeatures = 7
ground_truth_col = 0
# for the training data
train_X = train_data[:,1:nFeatures+1]   # The 7 features are in csv columns 2-8 [1:7]
train_truth = train_data[:,ground_truth_col]   # The 'survived' flag is in csv column 9 [8]
print('train_truth shape[0]', train_truth.shape[0])
# for the test data
test_X = test_data[:,1:nFeatures+1]   # The 7 features are in csv columns 2-8 [1:7]
test_truth = test_data[:,ground_truth_col]   # The 'survived' flag is in column 9 [8]
print('train_X', train_X.shape)
print('train_truth', train_truth.shape)
print('test_X', test_X.shape)
print('test_truth', test_truth.shape)
print()



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION #4:  Scale the values in the datasets
# -----------------------------------------------------------------------------------------------------------------
# As with the Brodie Weights model, there is a great difference in magnitude between the values of the various
# Features.  As before, a typical way to solve this is to individually scale the input Features, so they are all
# normalized (centered on zero with a standard deviation of 1).  (This means that after scaling they will mostly
# be between -1 and +1.)  The following code does this.  Each Feature column is scaled individually across
# all the Training Example rows.
#
# Note that we compute a scaling object (scaler_obj), which contains the derived scale factors.  It is calculated
# from just the Training Features.  Once we have it, it is applied to both the Training Features and the Test
# Features, since they both must be scaled identically.
#
# (Typically, after this is done, the scaling object is output and saved in a file for later use, since it must be
# applied to the Features of every subsequent input to the model.  In this program, we don't save the scaling
# object because there are no additional Examples to be run in some later program.)
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# SCALE EACH FEATURE TO BE CENTERED ON ZERO, WITH A STANDARD DEVIATION OF 1
scaler_obj = preprocessing.StandardScaler().fit(train_X)   # scaler_obj will scale each Feature (column) independently
train_X_scaled = scaler_obj.transform(train_X)  # scale each Training Feature (column)
test_X_scaled = scaler_obj.transform(test_X)  # scale each Test Feature (column)


print('train_X', train_X_scaled.shape)
print('train_truth', train_truth.shape)
print('test_X', test_X_scaled.shape)
print('test_truth', test_truth.shape)


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION #5:  The DNN Model
# -----------------------------------------------------------------------------------------------------------------
# Now we are ready to build our model.  We use Keras to define the layers:
#  - There is one Input Layer, which simply receives the 7 Features.
#  - There are 2 Hidden (Activation) Layers; both of which use ReLU activation functions.
#  - And 1 Output Layer, which uses a single sigmoid activation function, since we want the model to predict just
#    one outcome = the probability of survival.
# The specified optimizer is 'adam' which is basically a gradient descent modified by a momentum factor.
# The optimizer will minimize the loss, which will be calculated as 'binary_crossentropy' because we want only
# a single probability value representing the 2 (binary) outcomes:  survived or perished.
# The screen print shows the model has 120 weights to be trained.
#
#
# AI MODELING NOTE:  Why are there 120 weights?
# -- The Input Layer is connected to the 1st Hidden Layer by 56 weights.  This is because the Input Layer has 8
#    nodes (7 for the 7 input Features plus a Bias weight node), and the 1st Hidden Layer has the 7 nodes that
#    I specified.  Since each node of one layer is connected (Dense) to every node of the next layer by a weight,
#    there will be 8x7 = 56 weights between the two layers.
# -- The 1st Hidden Layer is connected to the 2nd Hidden Layer by 56 weights.  This is because the 1st Hidden
#    Layer has 8 nodes (the 7 specified nodes plus a Bias weight node), and the 2nd Hidden Layer has the 7 nodes
#    that I specified.  Since each node of one layer is connected (Dense) to every node of the next layer by a
#    weight, there will be 8x7 = 56 weights between the two layers.
# -- Finally, the 2nd Hidden Layer is connected to the Output Layer by 8 weights.  This is because the 2nd Hidden
#    Layer has 8 nodes (the 7 specified nodes plus a Bias weight node) and the sigmoid Output Layer has
#    1 node.  Since each node of one layer is connected (Dense) to every node of the next layer by a weight, there
#    will be 8x1 = 8 weights between the two layers.
# Therefore the total number of weights is 56 + 56 + 8 = 120 weights.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# BUILD THE TENSORFLOW MODEL
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[nFeatures,]))
model.add(keras.layers.Dense(nFeatures, activation='relu'))
model.add(keras.layers.Dense(nFeatures, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=["accuracy"])





# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION #6:  Run the model
# -----------------------------------------------------------------------------------------------------------------
# The TensorFlow 'fit' method is used to compute the best weights for the Training Examples.  The 120 weights
# are trained over 10 epochs.  Note that the sigmoid activation function outputs a probability of survival (between
# 0 and 1).  We will consider anything >= 0.5 as a prediction of survival.  The comparison of the Training Examples
# to their Ground Truth values are stored in the 'history' object for plotting.
#
# The Test Examples are also input to the fit method and the predictions from our evolving model (derived from the
# Training Examples) are compared against the matching Test Ground Truth values.  This gives a validation
# measure of how the model will do when data it has never seen before are input.  The comparison of the Test
# Examples to their Ground Truth values are also stored in the 'history' object for plotting.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# FIND BEST VALUES FOR THE 120 WEIGHTS
history = model.fit(train_X_scaled, train_truth, batch_size=1, epochs=10, validation_data=(test_X_scaled, test_truth))



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION #7:  Print/plot information about its accuracy
# -----------------------------------------------------------------------------------------------------------------
# To get a measure of the accuracy of the new model, the 'evaluate' method can be run on both the Training
# Examples and their matching Ground Truth values, and on the Test Examples and their matching Ground Truth
# values.  For each set of examples, the 'loss' shows the Cost of the errors (i.e. false positives and
# negatives).  Lower Cost is better.  The 'accuracy' shows how well the predicted probabilities matched the
# Ground Truth values.  Higher accuracy is better.
#
# For this model's Training data, after 10 epochs the cost was 0.44 (arbitrary units), and the prediction
# accuracy was about 79%.  This means given a Training Example passenger's Features, we can predict that
# passenger's survival with about 79% accuracy.  This give us a measure of whether our model is appropriate for
# the task at hand.
# For this model's Test Data (USING THE WEIGHTS DERIVED FROM THE TRAINING EXAMPLES) the cost was 0.49, and
# the prediction accuracy was about 77%.  This means given any passenger's Features, we can predict that
# passenger's survival with about a 77% accuracy.  This gives us a measure of whether our model will work well
# on entirely new data.
#
# We would expect the Training Examples' accuracy and Cost to be better than the Test Examples' accuracy and
# Cost, since the model weights were derived from the Training Examples!  Regardless, the accuracy and Cost
# for the Test Examples should be similar to those of the Training Examples, if the model we derived is any good.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION #8:  Print a Confusion Matrix on the Test Examples
# -----------------------------------------------------------------------------------------------------------------
# One way to see the relationship between cost and accuracy is to compute a Confusion Matrix.  It shows all the
# Testing Examples' outcomes collected into a square of 4 groupings for quick comparison:
#
#                        true negatives,    false negatives
#                        false positives,   true positives
#
# The Confusion Matrix is run on the Test Examples because those are the best indicators of how accurate the
# model is.
#
# Ideally, the "true" diagonal of 'true negatives' and 'true positives' (in this case, correct 'perished'
# and correct 'survived' predictions) should be far greater than the "false" diagonal of 'false positives' and
# 'false negatives' (in this case, incorrect 'survived' and incorrect 'perished' predictions).
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# PRINT A CONFUSION MATRIX FOR THE TEST EXAMPLES
probabilities = model.predict(test_X_scaled)   # the probability that the passengers survived
# convert the probabilities to predictions (0 or 1) for comparison with the ground truth
min_for_true = 0.5   # if the probability is >= 0.5, then assume the passenger survived
vector_int = np.vectorize(np.int)
predictions = vector_int(probabilities + min_for_true)    # The int truncates any fractional part of the sum
((n_true_negatives, n_false_negatives), (n_false_positives, n_true_positives)) \
    = confusion_matrix(test_truth, predictions)
print()
print()
print('Confusion Matrix for Survival on Test Examples (counts of passengers)')
print()
print('     Perished Correctly Predicted:   ', n_true_negatives, '  ',
    n_false_negatives, ' :Perished Incorrectly Predicted')
print('     Survival Incorrectly Predicted:  ', n_false_positives, '  ',
    n_true_positives, ' :Survival Correctly Predicted')
print()



# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# DISCUSSION #9:  Print some examples
# -----------------------------------------------------------------------------------------------------------------
# Now that we have derived a model and weights, we can use it to predict the outcomes of new passengers.  However,
# there are no new passengers, so here I've simply predicted the outcomes of the first 20 Test Examples.  The table
# shows the passenger data, the resulting prediction, and the actual Ground Truth outcome for comparison.
# The predictions should be correct about 77% of the time.
#
# Note that the model and its weights are relatively opaque.  It can predict who will survive, but it's difficult
# to know why!  This is a simple model with only 120 weights.  Yet it would be nearly impossible to translate those
# weights into an understanding of the relative importance of each input Feature.  Therefore, the model is rather
# like a black box.  Sadly, this opacity is typical of AI models.
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# PRINT SOME OF THE INPUT DATA EXAMPLES
print()
print()
print('Some Passenger Data, with Predictions and Actual Outcomes...')
print(' Age    Fare    Sex     SibSp     ParCh     Class   Embark   [Pred]Survived    [Actual]Survived')
for i in range(0, 20):
    print('{:4.0f}'.format(test_X[i,0]), '  ',
        '${:3.0f}'.format(test_X[i,1]), ' ',
        '{:3.0f}'.format(test_X[i,2]), '    ',
        '{:3.0f}'.format(test_X[i, 3]), '     ',
        '{:3.0f}'.format(test_X[i, 4]), '     ',
        '{:3.0f}'.format(test_X[i, 5]), '    ',
        '{:3.0f}'.format(test_X[i, 6]), '      ',
        '{}'.format(predictions[i]==1), '         ',
        '{}'.format(test_truth[i]==1)
    )
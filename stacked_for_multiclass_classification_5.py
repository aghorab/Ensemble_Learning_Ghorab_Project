# example of saving sub-models for later use in a stacking ensemble
from sklearn.datasets import make_blobs
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from os import makedirs

# To keep this example simple, we will use multiple instances of the same model
# as level-0 or sub-models in the stacking ensemble.
#
# We will also use a holdout validation dataset to train the level-1 or
# meta-learner in the ensemble.
#
# In this example, we will train multiple sub-models and save them to
# file for later use in our stacking ensembles.
#

# fit model on dataset
def fit_model(trainX, trainy):
    # define model
    model1 = Sequential()
    model1.add(Dense(25, input_dim=2, activation='relu'))
    model1.add(Dense(3, activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model1.fit(trainX, trainy, epochs=500, verbose=0)
    return model1


# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)

# one hot encode output variable
# Converts a class vector (integers) to binary class matrix.
# [2 0 2 ...] to [[0. 0. 1.] [1. 0. 0.] [0. 0. 1.] ...]
y = to_categorical(y)


# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
print(trainX.shape, testX.shape)

# create directory for models
makedirs('models')

# fit and save models
n_members = 5
for i in range(n_members):
    # fit model
    model = fit_model(trainX, trainy)   # call the method above
    # save model
    filename = 'models/model_' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)

# The problem is a multi-class classification problem,
# and we will model it using a softmax activation function on the output layer.
# This means that the model will predict a vector with three elements with the probability
# that the sample belongs to each of the three classes.
# Therefore, we must one hot encode the class values before we split
# the rows into the train and test datasets. We can do this using the Keras
# to_categorical() function.

# The model will expect samples with two input variables.
# The model then has a single hidden layer with 25 nodes and a rectified linear activation function,
# then an output layer with three nodes to predict the probability
# of each of the three classes and a softmax activation function.

# Because the problem is multi-class,
# we will use the categorical cross entropy loss function to optimize the model and the
# efficient Adam flavor of stochastic gradient descent.







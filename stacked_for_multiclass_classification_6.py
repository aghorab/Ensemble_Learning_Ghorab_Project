# stacked generalization with linear meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from numpy import dstack


# Separate Stacking Model
# We can now train a meta-learner that will best combine the predictions
# from the sub-models and ideally perform better than any single sub-model.
#
# The first step is to load the saved models.

# Next, we can train our meta-learner. This requires two steps:
# Prepare a training dataset for the meta-learner.
# Use the prepared training dataset to fit a meta-learner model.
# We will prepare a training dataset for the meta-learner by providing
# examples from the test set to each of the sub-models and collecting the predictions.
# In this case, each model will output three predictions for each example
# for the probabilities that a given example belongs to each of the three classes.
# Therefore, the 1,000 examples in the test set will result in five arrays with the
# shape [1000, 3].

# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'models/model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# As input for a new model, we will require 1,000 examples with some number
# of features. Given that we have five models and each model makes three predictions per example,
# then we would have 15 (3 x 5) features for each example provided to the submodels.
# We can transform the [1000, 5, 3] shaped predictions from the sub-models into a [1000, 15]
# shaped array to be used to train a meta-learner using the reshape() NumPy function and
# flattening the final two dimensions. The stacked_dataset() function implements this step.

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities 5*3 = 15] = [1000, 15]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX


# meta-learner

# In this case, we will train a simple logistic regression algorithm from the scikit-learn library.
# Logistic regression only supports binary classification,
# although the implementation of logistic regression in scikit-learn in
# the LogisticRegression class supports multi-class classification (more than two classes)
# using a one-vs-rest scheme. The function fit_stacked_model() below will prepare the training
# dataset for the meta-learner by calling the stacked_dataset() function, then fit a logistic
# regression model that is then returned.
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = LogisticRegression()
    # stackedX = [1000, 15] every data item is predicted 5 times each prediction contains 3 values
    model.fit(stackedX, inputy)  ############## inputs are the predictions of the 5 models
    return model


# make a prediction with the stacked model
def stacked_prediction(members, stacked_model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = stacked_model.predict(stackedX)
    return yhat


# generate 2d classification dataset
X, y = make_blobs(n_samples=1100, centers=3, n_features=2, cluster_std=2, random_state=2)

# split into train and test
n_train = 100
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
print(trainX.shape, testX.shape)

# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# evaluate standalone models on test dataset
for model in members:
    testy_enc = to_categorical(testy)
    _, acc = model.evaluate(testX, testy_enc, verbose=0)
    print('Model Accuracy: %.3f' % acc)

# fit stacked model using the ensemble ##################### here we engage the meta-learner
ensemble_model = fit_stacked_model(members, testX, testy)

# evaluate model on test set
yhat = stacked_prediction(members, ensemble_model, testX)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)

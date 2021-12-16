# The number of features that is randomly sampled for each split
# point is perhaps the most important feature to configure for random forest.

# It is set via the max_features argument and defaults to the square
# root of the number of input features. In this case, for our test dataset,
# this would be sqrt(20) or about four features.

# The example below explores the effect of the number of features randomly
# selected at each split point on model accuracy. We will try values from 1 to 7 and would expect a
# small value, around four, to perform well based on the heuristic.


# explore random forest number of features effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot


# get the dataset
def get_dataset():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
    return X, y


# get a list of models to evaluate
def get_models():
    models = dict()
    # explore number of features from 1 to 7
    for i in range(1, 8):
        models[str(i)] = RandomForestClassifier(max_features=i)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X, y)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# مع تكرار التجربة أحيانا يكون عدد الخصائص الاقل افضل واحيانا الاكبر افضل لكن في الغالب 4 او 5 هو الافضل
# وهي قيمة الجذر التربيعي لعدد الخصائص
# Q. How many features should be chosen at each split point?
#
# The best practice is to test a suite of different values and discover what works best for your dataset.
#
# As a heuristic, you can use:
#
# Classification: Square root of the number of features.
# Regression: One third of the number of features.

# Import scikit-learn dataset library
import math

from sklearn import datasets
import pandas as pd
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = datasets.load_iris()

# print the names of the four features
print("All Features: " + str(iris.feature_names))

# print the label species(setosa, versicolor,virginica)
print("Target Names: " + str(iris.target_names))

# print the iris data (top 5 records)
# print(iris.data[0:5])

# print the iris labels (0:setosa, 1:versicolor, 2:virginica)
# print(iris.target)

# Creating a DataFrame of given iris dataset.


data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})
print(data.head())

X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

# Create a Gaussian Classifier
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                             oob_score=False, random_state=None, verbose=0,
                             warm_start=False)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# ########################################
# ######## Feature importance    #########
# ########################################
# ########################################


feature_imp = pd.Series(clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
print("\nFeature Importance Values:")
for index, val in feature_imp.iteritems():
    print(f'{index}\t: {val}')
print("Total Sum: " + str(math.fsum(feature_imp)))

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

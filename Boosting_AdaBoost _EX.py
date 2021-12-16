from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ad = AdaBoostClassifier(n_estimators=100, learning_rate=0.03)

ad.fit(X_train, y_train)
score = ad.score(X_test, y_test)

print(score)

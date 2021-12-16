# https://github.com/YashK07/Stacking-Ensembling/blob/main/Stacking%20in%20Machine%20Learning.ipynb
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
import shutup




shutup.please()

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message="the default evaluation metric")
warnings.filterwarnings("ignore", message="XGBoost")
warnings.filterwarnings("ignore", message="learner.cc:1115: Starting in XGBoost 1.3.0")
warnings.filterwarnings(action='once')

df = datasets.load_breast_cancer()
X = pd.DataFrame(columns=df.feature_names, data=df.data)
y = df.target
print(X.head())
print(X.isnull().sum())

print(df.target.shape)

target = {'target': df.target}
y = pd.DataFrame(data=target)
print(y.value_counts())
# The data looks balanced, so we will choose accuracy as our metric.
# 1 - Benign
# 0 - Malignant


y = y['target']
print(X.describe())
# Takeaways :
# The data doesnot require any preprocessing.
# EDA is not as such required so we move ahead to the modeling part.


dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
xgb = xgboost.XGBClassifier()

clf = [dtc, rfc, knn, xgb]
for algo in clf:
    score = cross_val_score(algo, X, y, cv=5, scoring='accuracy')
    print("The accuracy score of {} is:".format(algo), score.mean())

print('########################################### Stacking ######################################')
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
xgb = xgboost.XGBClassifier()
clf = [('dtc', dtc), ('rfc', rfc), ('knn', knn), ('xgb', xgb)]  # list of (str, estimator)

lr = LogisticRegression()
stack_model = StackingClassifier(estimators=clf, final_estimator=lr)
score = cross_val_score(stack_model, X, y, cv=5, scoring='accuracy')
print("The accuracy score of is:", score.mean())

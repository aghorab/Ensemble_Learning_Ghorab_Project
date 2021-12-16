# make a prediction with a stacking ensemble
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)

print(f'X = {X[0]} ==> y={y[0]}')
print('===============================================================================')
# define the base models
level0 = list()
level0.append(('knn', KNeighborsRegressor()))
level0.append(('cart', DecisionTreeRegressor()))
level0.append(('svm', SVR()))
# define meta learner model
level1 = LinearRegression()
# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(X, y)
# make a prediction for one example
data = [[0.4212206, -0.5543507, 1.34543718, -0.57054047, -0.72480487, 1.05648449, 0.77744852,
         0.07361796, 0.88398267, 2.043523157, 1.01902732, 0.11227799, 0.94218853, 0.26741783, 0.91458143,
         -0.72759572, 1.08842814, -5.61450942, -0.69387293, 8.69169009]]

yhat = model.predict(data)
print(f'\ndata ={data[0]}  \nPredicted Value: %.3f' % (yhat))









#
# Regression analysis consists of a set of machine learning methods that allow us
# to predict a continuous outcome variable (y) based on the value of one or multiple predictor
# variables (x). Briefly, the goal of regression model is to build a mathematical equation that defines y as
# a function of the x variables.

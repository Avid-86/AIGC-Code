from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#1.get datasets
iris = load_iris()
#2.data process
#standard datasets,do not need data process, just split datasets?
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=22,test_size=0.2)
#3.feature process
#1)instantiate a transformer
transfer = StandardScaler()
#2)invoke function "fit_transform "
transfer.fit_transform(x_train)
transfer.fit_transform(x_test)
#4.model training
estimator = KNeighborsClassifier(n_neighbors= 1 )
#invoke cross validation grid search model
param_grid={"n_neighbors":[1,3,5,7,9]}
estimator = GridSearchCV(estimator,param_grid=param_grid,cv=10)
estimator.fit(x_train,y_train)
#5.model evaluation
#1)output predicted value
y_pre = estimator.predict(x_test)
print("predicted value is: \n",y_pre)
print("comparison of accurate and predicted values:\n",y_pre==y_test)
#2)output accurate value
ret = estimator.score(x_test,y_test)
print("accurate value is: \n",ret)
print("best model:\n",estimator.best_estimator_)
print("best result:\n",estimator.best_score_)
print("integral model result:\n",estimator.cv_results_)






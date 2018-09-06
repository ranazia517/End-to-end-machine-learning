# Classification Project: Sonar rocks or mines

# Load libraries
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pandas import read_csv
from pandas import set_option
from sklearn import metrics
#import numpy as np
#from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Load dataset
filename = 'Data_baseline_full_Axivity_Single_Intermitent_Silvia_python_Axivity_4.csv'
names = ['StepTimeGR', 'StepTimeVariabilitySDGR', 'StepTimeAsymmetryGR', 'StanceTimeGR', 'Label']
dataset = read_csv(filename, names=names)



# Summarize Data

# Descriptive statistics
# shape
print(dataset.shape)
# types
set_option('display.max_rows', 500)
print(dataset.dtypes)
# head
set_option('display.width', 100)
print(dataset.head(20))
# descriptions, change precision to 3 places
set_option('precision', 3)
print(dataset.describe())
# class distribution
print(dataset.groupby('Label').size())


# Data visualizations

## histograms
#dataset.hist()
#pyplot.show()
## density
#dataset.plot(kind='density', subplots=True, layout=(5,2), sharex=False, legend=False)
#pyplot.show()
## box and whisker plots
## dataset.plot(kind='box', subplots=True, layout=(w,8), sharex=False, sharey=False)
## pyplot.show()
#
## scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()
## correlation matrix
#fig = pyplot.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
#fig.colorbar(cax)
#pyplot.show()

# Prepare Data

# Split-out validation dataset
array = dataset.values
X = array[:,0:4].astype(float)
Y = array[:,4]
validation_size = 0.10
seed = 5
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Evaluate Algorithms

# Test options and evaluation metric
num_folds = 10
seed = 5
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

    
    
## Compare Algorithms
#fig = pyplot.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#pyplot.boxplot(results)
#ax.set_xticklabels(names)
#pyplot.show()


# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
#results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, shuffle=False, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print(name)
    print(cv_results)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
#    for train_index, test_index in kfold.split(X_train):
##                  print("TRAIN:", train_index, "TEST:", test_index)
#                  x_train, x_test = X_train[train_index], X_train[test_index]
#                  y_train, y_test = Y_train[train_index], Y_train[test_index]
#                  print(name)
#                  print(cv_results)
#                  model.fit(x_train, y_train)
#                  predictions = model.predict(x_test)
#                  fpr,tpr,thres=metrics.roc_curve(y_test,predictions)
#                  print ('senstivity', tpr)
#                  print ('specificity', 1-fpr)
#                  print ('Area under the curve', metrics.auc(fpr,tpr))
##                  plt.plot(fpr, tpr)
##                  plt.show()
#                  print(accuracy_score(y_test, predictions))
#                  print(confusion_matrix(y_test, predictions))
#                  print(classification_report(y_test, predictions))
      
#
#    
#
#    
#
#
## Compare Algorithms
#fig = pyplot.figure()
#fig.suptitle('Scaled Algorithm Comparison')
#ax = fig.add_subplot(111)
#pyplot.boxplot(results)
#ax.set_xticklabels(names)
#pyplot.show()
#
#
## Tune scaled KNN
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#neighbors = [1,3,5,7,9,11,13,15,17,19,21]
#param_grid = dict(n_neighbors=neighbors)
#model = KNeighborsClassifier()
#kfold = KFold(n_splits=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))
#
## Tune scaled Logistic regression
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#C = [0.001,0.01,0.1,1,10,100,1000]
#param_grid = dict(C=C)
#model = LogisticRegression()
#kfold = KFold(n_splits=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))
#
### Tune scaled Random Forest
##scaler = StandardScaler().fit(X_train)
##rescaledX = scaler.transform(X_train)
### Number of trees in random forest
##n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
### Number of features to consider at every split
##max_features = ['auto', 'sqrt']
### Maximum number of levels in tree
##max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
##max_depth.append(None)
### Minimum number of samples required to split a node
##min_samples_split = [2, 5, 10]
### Minimum number of samples required at each leaf node
##min_samples_leaf = [1, 2, 4]
### Method of selecting samples for training each tree
##bootstrap = [True, False]
##param_grid = dict(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap)
##model = RandomForestClassifier()
##kfold = KFold(n_splits=num_folds, random_state=seed)
##grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
##grid_result = grid.fit(rescaledX, Y_train)
##print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
##means = grid_result.cv_results_['mean_test_score']
##stds = grid_result.cv_results_['std_test_score']
##params = grid_result.cv_results_['params']
##for mean, stdev, param in zip(means, stds, params):
##    print("%f (%f) with: %r" % (mean, stdev, param))
#
#
#
## Tune scaled SVM
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0, 10, 20]
#kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
#param_grid = dict(C=c_values, kernel=kernel_values)
#model = SVC()
#kfold = KFold(n_splits=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))
#
#
#
#
## ensembles
#ensembles = []
##ensembles.append(('AB', AdaBoostClassifier()))
##ensembles.append(('GBM', GradientBoostingClassifier()))
#ensembles.append(('RF', RandomForestClassifier(n_estimators=1000)))
#ensembles.append(('ET', ExtraTreesClassifier()))
#results = []
#names = []
#for name, model in ensembles:
##	kfold = KFold(n_splits=num_folds, random_state=seed)
##	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
##	results.append(cv_results)
##	names.append(name)
##	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
##	print(msg)
#    kfold = KFold(n_splits=num_folds, shuffle=False, random_state=None)
#    for train_index, test_index in kfold.split(X_train):
##                  print("TRAIN:", train_index, "TEST:", test_index)
#                  x_train, x_test = X_train[train_index], X_train[test_index]
#                  y_train, y_test = Y_train[train_index], Y_train[test_index]
#                 
#                  print(name)
#                  print(cv_results)
#                  model.fit(x_train, y_train)
#                  predictions = model.predict(x_test)
#                  fpr,tpr,thres=metrics.roc_curve(y_test,predictions)
#
#                  print ('senstivity', tpr)
#                  print ('specificity', 1-fpr)
#                  print ('Area under the curve', metrics.auc(fpr,tpr))
##                  plt.plot(fpr, tpr)
##                  plt.show()
#                  print(accuracy_score(y_test, predictions))
#                  print(confusion_matrix(y_test, predictions))
#                  print(classification_report(y_test, predictions))
                  
## Compare Algorithms
#fig = pyplot.figure()
#fig.suptitle('Ensemble Algorithm Comparison')
#ax = fig.add_subplot(111)
#pyplot.boxplot(results)
#ax.set_xticklabels(names)
#pyplot.show()
#
#
#
### Finalize Model
##
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = LogisticRegression()
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Logistic Regression")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
#
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = LinearDiscriminantAnalysis()
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Linear Discriminant Analysis")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
#
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = KNeighborsClassifier()
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("K nearest Neighbour")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = DecisionTreeClassifier()
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Decision Tree")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = GaussianNB()
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Guasian Basic Function")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
#
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = SVC(kernel='linear', C=0.3)
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Support Vector Machine")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
#
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = RandomForestClassifier(n_estimators=1000)
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Randome forest Classifier")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
#
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = GradientBoostingClassifier()
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Gradient Boosting Classifier")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
#
## Finalize Model
#
## prepare the model
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#model = ExtraTreesClassifier()
#model.fit(rescaledX, Y_train)
## estimate accuracy on validation dataset
#rescaledValidationX = scaler.transform(X_validation)
#predictions = model.predict(rescaledValidationX)
#print("Extra Tree Classifier")
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#fpr,tpr,thres=metrics.roc_curve(Y_validation , predictions)
#print ('senstivity', tpr)
#print ('specificity', 1-fpr)
#print ('Area under the curve', metrics.auc(fpr,tpr))
#
#
#
#

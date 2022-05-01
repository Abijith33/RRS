# Check the versions of libraries

# Python version
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from pandas import read_csv
import sklearn
import pandas as pd
import matplotlib
import numpy
import scipy
import sys


# Load libraries
...
data = pd.read_csv(r'D:\Sem 7\Project\Demo\Data2.csv')
df = pd.DataFrame(data, columns=['Name', 'Admission Number', '10th', '12th/ Diploma', 'CGPA',
                  'C/C++', 'Java', 'Python', 'Other', 'Placement Status', 'Package (in LPA)'])
print(df)
...
# # descriptions
# print(data.describe())
# # class distribution
# print(data.groupby('Placement Status').size())
# box and whisker plots
# data.plot(kind='box', subplots=True, layout=(
#     9, 9), sharex=False, sharey=False)
# pyplot.show()
# histograms
# data.hist()
# pyplot.show()
# scatter plot matrix
# scatter_matrix(data)
# pyplot.show()
...
# Split-out validation dataset
array = data.values
X = array[:, range(2, 10)]
y = array[:, 7]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.25, random_state=0)
...
# Spot Check Algorithms
""" models = []
models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())) """
""" make predictions """


# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

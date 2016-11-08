import numpy as np
from sklearn.model_selection import cross_val_score

FILE_PATH = 'preprocessed_train.csv'

print "Loading dataset from file %s " % (FILE_PATH)
dataset = np.genfromtxt(FILE_PATH, delimiter=',',skip_header=1)
print "Dataset loaded..."

print "Separating dataset in training_data and labels_data"

# Get all columns but the last
training_data = dataset[:, :(dataset.shape[1] - 1) ]

# Get the last column only - target
target_data   = dataset[:, dataset.shape[1] - 1 ]

# SVM takes a long time to train
# and it not generates really good results
# from sklearn import svm
# clf = svm.SVC()

# The standard MPL classifier get as good as random
# maybe it just needs tuning
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier()

# Best result for now, AUC average: 82.0, std: 0.1
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=15, n_estimators=50)

# from sklearn.ensemble import ExtraTreesClassifier
# clf = ExtraTreesClassifier(max_depth=15, n_estimators=50)

# Like the MPL it's as good as random
# from sklearn.naive_bayes import GaussianNB
# clf   = GaussianNB()

scores = cross_val_score(clf, training_data, target_data, cv=10, scoring='roc_auc')

print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# import matplotlib.pyplot as plt
# plt.plot(scores_means)
# plt.show()

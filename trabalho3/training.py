import pickle as pk
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import ttest_rel

FILE_PATH = 'preprocessed_train.csv'

print "Loading dataset from file %s" % (FILE_PATH)
preprocessed_dataset = np.genfromtxt(FILE_PATH, delimiter=',')
# original_dataset     = np.genfromtxt('train_file.csv', delimiter=',', skip_header=1)
print "Datasets loaded..."

print "Separating dataset in training_data and labels_data"

def train_model(dataset):
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
    # clf = MLPClassifier(learning_rate='adaptive',activation='logistic')

    # Best result for now, AUC average: 82.0, std: 0.1
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=15,
        n_estimators=100,
        max_features=None,
        min_samples_leaf=50,
        n_jobs=-1
    )

    # from sklearn.ensemble import ExtraTreesClassifier
    # clf = ExtraTreesClassifier(max_depth=15, n_estimators=50)

    # Like the MPL it's as good as random
    # from sklearn.naive_bayes import GaussianNB
    # clf   = GaussianNB()

    # Best result for now, AUC average: 82.0, std: 0.1
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(max_depth=15, n_estimators=50)

    clf.fit(training_data, target_data)
    return clf

def test_model_with_cross_validation(dataset,clf):
    # Get all columns but the last
    training_data = dataset[:, :(dataset.shape[1] - 1) ]

    # Get the last column only - target
    target_data   = dataset[:, dataset.shape[1] - 1 ]

    k = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf, training_data, target_data, cv=k, scoring='roc_auc')

    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf = train_model(preprocessed_dataset)
test_model_with_cross_validation(preprocessed_dataset, clf)
model_file = open('model.pickle',"wb")
pk.dump(clf, model_file)

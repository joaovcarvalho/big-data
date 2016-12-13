import pickle as pk
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import ttest_rel

FILE_PATH = 'preprocessed_train.csv'

print "Loading dataset from file %s" % (FILE_PATH)
dataset = np.genfromtxt(FILE_PATH, delimiter=',')
# original_dataset     = np.genfromtxt('train_file.csv', delimiter=',', skip_header=1)
print "Datasets loaded..."

print "Separating dataset in training_data and labels_data"

# Get all columns but the last
training_data = dataset[:, :(dataset.shape[1] - 1) ]

# Get the last column only - target
target_data   = dataset[:, dataset.shape[1] - 1 ]

def model():
    # SVM takes a long time to train
    # and it not generates really good results
    # from sklearn import svm
    # clf = svm.SVC()

    # The standard MPL classifier get as good as random
    # maybe it just needs tuning
    # from sklearn.neural_network import MLPClassifier
    # clf = MLPClassifier(learning_rate='adaptive',activation='logistic')

    # Best result for now, AUC average: 82.0, std: 0.1
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.02,
        max_features=0.3,
        max_depth=4,
        min_samples_leaf=17
    )

    # from keras.wrappers.scikit_learn import KerasClassifier
    # def create_model():
    #     from keras.models import Sequential
    #     model = Sequential()
    #
    #     from keras.layers import Dense, Activation
    #     # create model
    #     model = Sequential()
    #     model.add(Dense(120, input_dim=training_data.shape[1], init='glorot_normal', activation='relu'))
    #     # model.add(Dense(1024, init='normal', activation='relu'))
    #     # model.add(Dense(256, init='normal', activation='relu'))
    #     # model.add(Dense(64, init='normal', activation='relu'))
    #     # model.add(Dense(32, init='normal', activation='relu'))
    #     model.add(Dense(1, init='glorot_normal', activation='softmax'))
    #     # Compile model
    #     # from keras.optimizers import SGD
    #     # sgd = SGD(lr=0.01, decay=1e-6)
    #     from keras.optimizers import Adam
    #     adam = Adam(lr=1.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #     model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])
    #     return model

    # model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)

    # clf = KerasClassifier(build_fn=create_model, nb_epoch=1, batch_size=128, verbose=1)

    # from sklearn.ensemble import ExtraTreesClassifier
    # clf = ExtraTreesClassifier(max_depth=15, n_estimators=50)

    # Like the MPL it's as good as random
    # from sklearn.naive_bayes import GaussianNB
    # clf   = GaussianNB()

    # Best result for now, AUC average: 82.0, std: 0.1
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(max_depth=15, n_estimators=50)

    return clf

def test_model_with_cross_validation(dataset,clf):
    # Get all columns but the last
    training_data = dataset[:, :(dataset.shape[1] - 1) ]

    # Get the last column only - target
    target_data   = dataset[:, dataset.shape[1] - 1 ]

    k = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(clf,
        training_data,
        target_data,
        n_jobs=-1,
        verbose=1,
        cv=k,
        scoring='roc_auc')

    print ("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

clf = model()
clf.fit(training_data,target_data)
# from sklearn.grid_search import GridSearchCV
# param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
#               'max_depth': [4,6],
#               'min_samples_leaf': [3,5,9,17],
#               'max_features': [1.0, 0.3, 0.1]
# }
# {'max_features': 0.3, 'learning_rate': 0.1, 'max_depth': 6, 'min_samples_leaf': 17}
# gs_cv = GridSearchCV(clf, param_grid, scoring='roc_auc', verbose=1, n_jobs=-1).fit(training_data, target_data)
# print gs_cv.best_params_
model_file = open('model.pickle',"wb")
pk.dump(clf, model_file)
# test_model_with_cross_validation(dataset, clf)

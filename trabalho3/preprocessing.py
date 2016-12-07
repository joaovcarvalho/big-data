import pickle as pk
import numpy as np
from sklearn import preprocessing
from preprocessing_library import *

train_dataset = np.genfromtxt('train_file.csv', delimiter=',',skip_header=1)
test_dataset = np.genfromtxt('test_file.csv', delimiter=',',skip_header=1)

# Get the last column only - target
target_data   = train_dataset[:, train_dataset.shape[1] - 1 ]

# Get all columns but the last
train_dataset = train_dataset[:, :(train_dataset.shape[1] - 1) ]

dataset = np.concatenate((train_dataset, test_dataset), axis=0)

before_processing_columns = dataset.shape[1]

preprocessing_steps = {}

scaler  = preprocessing.StandardScaler()

# Remove zero variance columns
dataset = removeZeroVarianceColumns(dataset)

# Get second column that is equal to first
# we just need to remove one of the columns
dataset = removeColumnsThatAreEqual(dataset)

# dataset = removeCorrelatedColumns(dataset)

categorical_indexes = get_categorical_features(dataset)
dataset = encode_test_dataset(dataset, categorical_indexes)

# Add last column for targets
# train_dataset[:,-1] = target_data
train_dataset = dataset[:train_dataset.shape[0]]
train_dataset = np.column_stack( (train_dataset, target_data) )

# Get only a subsample of the train_dataset
# so we can iterate faster for testing
# train_dataset = train_dataset[:(10000),:]

after_processing_columns = dataset.shape[1]
print ("Columns removed: %d" % (before_processing_columns - after_processing_columns))
print "Columns left: %d" %(after_processing_columns)

np.savetxt('preprocessed_train.csv',train_dataset,delimiter=",")
print "Train dataset preprocessed with success"

test_dataset = dataset[ -6000:]

np.savetxt('preprocessed_test.csv',test_dataset,delimiter=",")
print "Test dataset preprocessed with success"

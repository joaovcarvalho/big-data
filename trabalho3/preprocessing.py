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

before_processing_columns = train_dataset.shape[1]

preprocessing_steps = {}

scaler  = preprocessing.StandardScaler()

# Remove zero variance columns
train_dataset, preprocessing_steps['zero_variance_columns'] = removeZeroVarianceColumns(train_dataset)

# Get second column that is equal to first
# we just need to remove one of the columns
train_dataset, preprocessing_steps['columns_that_are_equal'] = removeColumnsThatAreEqual(train_dataset)

# train_dataset,c = removeCorrelatedColumns(train_dataset)

train_dataset, preprocessing_steps['categorical_indexes'] = encodeCategoricalFeatures(train_dataset)

print train_dataset.shape
train_dataset = scaler.fit_transform(train_dataset)
print train_dataset.shape

# Add last column for targets
# train_dataset[:,-1] = target_data
train_dataset = np.column_stack( (train_dataset, target_data) )


# Get only a subsample of the train_dataset
# so we can iterate faster for testing
# train_dataset = train_dataset[:(10000),:]

after_processing_columns = train_dataset.shape[1]
print ("Columns removed: %d" % (before_processing_columns - after_processing_columns))
print "Columns left: %d" %(after_processing_columns)

# p_file = open('preprocessing.pickle',"wb")
# pk.dump(preprocessing_steps, p_file)

np.savetxt('preprocessed_train.csv',train_dataset,delimiter=",")
print "Train dataset preprocessed with success"

# preprocessing test data
test_dataset = np.delete(test_dataset,preprocessing_steps['zero_variance_columns'],1)
test_dataset = np.delete(test_dataset,preprocessing_steps['columns_that_are_equal'],1)
test_dataset = encode_test_dataset(test_dataset,preprocessing_steps['categorical_indexes'])
test_dataset = scaler.transform(test_dataset)

np.savetxt('preprocessed_test.csv',test_dataset,delimiter=",")
print "Test dataset preprocessed with success"

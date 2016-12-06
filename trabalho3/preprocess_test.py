import numpy as np
from sklearn import preprocessing
from preprocessing_library import *
import pickle

dataset = np.genfromtxt('test_file.csv', delimiter=',',skip_header=1)
p_file = open('preprocessing.pickle',"rb")
preprocessing_steps = pickle.load(p_file)

dataset = np.delete(dataset,preprocessing_steps['zero_variance_columns'],1)
dataset = np.delete(dataset,preprocessing_steps['columns_that_are_equal'],1)
dataset = encode_test_dataset(dataset,preprocessing_steps['categorical_indexes']);
scaler  = preprocessing_steps['scaler']
preprocessed_dataset = scaler.transform(dataset)

np.savetxt('preprocessed_test.csv',preprocessed_dataset,delimiter=",")
print "Dataset preprocessed with success"

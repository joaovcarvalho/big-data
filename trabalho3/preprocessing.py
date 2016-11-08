import numpy as np
from sklearn import preprocessing
from optparse import OptionParser
from preprocessing_library import *

parser = OptionParser()
parser.add_option("-i", "--input_file", dest="input_file",
                  help="Input dataset", metavar="FILE")
parser.add_option("-o", "--output_file", dest="output_file",
                  help="Output dataset after preprocessed", metavar="FILE")
parser.add_option("-s", "--save",
                  dest="store_steps", default=True,
                  help="Save preprocessing steps")

(options, args) = parser.parse_args()

SCALER_FILE_NAME = "scaler.pickle"
DELETED_COLUMNS_FILE_NAME = "columns_removed.pickle"

dataset = np.genfromtxt(options.input_file, delimiter=',',skip_header=1)

# Get the last column only - target
target_data   = dataset[:, dataset.shape[1] - 1 ]

# Get all columns but the last
dataset = dataset[:, :(dataset.shape[1] - 1) ]

before_processing_columns = dataset.shape[1]

# Remove zero variance columns
dataset = removeZeroVarianceColumns(dataset)

# Get second column that is equal to first
# we just need to remove one of the columns
dataset = removeColumnsThatAreEqual(dataset)

dataset = removeCorrelatedColumns(dataset)

scaler  = preprocessing.RobustScaler()
dataset = scaler.fit_transform(dataset)

# Add last column for targets
dataset[:,-1] = target_data

dataset = removeWeaklyCorrelatedWithClassColumns(dataset)

# Get only a subsample of the dataset
# so we can iterate faster for testing
# dataset = dataset[:(30000),:]

after_processing_columns = dataset.shape[1]
print ("Columns removed: %d" % (before_processing_columns - after_processing_columns))
print "Columns left: %d" %(after_processing_columns)

np.savetxt(options.output_file,dataset,delimiter=",")
print "Dataset preprocessed with success"

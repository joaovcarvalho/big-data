import pandas as pd
import numpy as np
import scipy.stats as stats
import sklearn.preprocessing as preprocessing

# Remove duplicates like (1,2) and (2,1)
def removeDuplicates(data):
    for t1 in data:
        for t2 in data:
            if t1 == t2[::-1]:
                data.remove(t2)
    return data

def encode_test_dataset(dataset,categorical_indexes):
    df = pd.DataFrame(data=dataset)
    for column in categorical_indexes:
        just_dummies = pd.get_dummies(df[column])
        df = pd.concat([df, just_dummies], axis=1)
        df.drop([column], inplace=True, axis=1)
    return df.as_matrix()

def encodeCategoricalFeatures(dataset):
    categorical_indexes = []
    for index,column in enumerate(dataset.T):
        uniques = np.unique(column)
        if uniques.size < 20 and np.all(column >= 0):
            categorical_indexes.append(index)

    df = pd.DataFrame(data=dataset)
    for column in categorical_indexes:
        just_dummies = pd.get_dummies(df[column])
        df = pd.concat([df, just_dummies], axis=1)
        df.drop([column], inplace=True, axis=1)

    return df.as_matrix(),categorical_indexes

# Return tuples of columns that are equal
def removeColumnsThatAreEqual(dataset, index = 1):
    columns = []
    # For each pair of columns check if all elements in both
    # columns are equal
    for i1,c1 in enumerate(dataset.T):
        for i2,c2 in enumerate(dataset.T):
            if i1 != i2:
                if np.all(c1 == c2):
                    columns.append( (i1,i2) )

    columns = removeDuplicates(columns)

    columns = [ x[index] for x in columns]

    dataset = np.delete(dataset, columns,1)
    return dataset, columns

# Returns column index that have zero variance
def removeZeroVarianceColumns(dataset):
    # Get all columns index that have variance 0.0
    zero_variance_columns = [index
            for index
            in range(dataset.shape[1])
            if np.var(dataset[:,index]) == 0.0
            ]
    dataset = np.delete(dataset,zero_variance_columns,1)
    return dataset, zero_variance_columns


def removeCorrelatedColumns(dataset, rho_min=0.9, p_value_max=0.05, index_to_remove=1):
    # Calculate Spearman correlation
    # If we have a high correlation and low p-value
    # then we can remove one of the columns
    rho, p_value = stats.spearmanr(dataset)
    correlated_columns = []

    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            if i != j:
                if rho[i,j] > rho_min and p_value[i,j] < p_value_max:
                    correlated_columns.append((i,j))
                    # print "Rho: %f, p-value: %f of columns: (%d,%d)" % (rho[i,j], p_value[i,j],i,j)

    correlated_columns = removeDuplicates(correlated_columns)
    correlated_columns = sorted(set( [ x[index_to_remove] for x in correlated_columns] ))

    dataset = np.delete(dataset, correlated_columns,1)
    return dataset, correlated_columns

def removeWeaklyCorrelatedWithClassColumns(dataset, rho_min=10e-3, p_value_max=0.05):
    rho, p_value = stats.spearmanr(dataset)

    weakly_correlated = []
    for i in range(dataset.shape[1]):
        if abs(rho[i, -1]) < rho_min and p_value[i, -1] < p_value_max:
            weakly_correlated.append(i)

    dataset = np.delete(dataset, weakly_correlated, 1)
    return dataset, weakly_correlated

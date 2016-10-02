import csv
import json
import numpy as np
import scipy.stats as stats
#import matplotlib.pyplot as plt

DICT_KEY = " click_on"

def answerToNumber(x):
    if x == 'yes':
        return 1;
    else:
        return 0;

def get_property_from_dataset(dataset, property_key):
    result = []
    for row in dataset:
        result.append( row[property_key])
    return result

def getClicksFromFile(file_path):
    with open(file_path, 'rb') as csvfile:
        dataset_sample_a = csv.DictReader(csvfile, delimiter=';', quotechar='|')

        clicks = np.array(get_property_from_dataset(dataset_sample_a, DICT_KEY));

        clicks =  map(answerToNumber, clicks);
        return clicks

clicks_sample_a = getClicksFromFile('amostra_A_click.csv');
clicks_sample_b = getClicksFromFile('amostra_B_click.csv');

ks,p_value = stats.ks_2samp(clicks_sample_a, clicks_sample_b)
print "Ks: " + str(ks)
print "P-value: " + str(p_value)

print "Sum Sample A: " + str( np.sum(clicks_sample_a) )
print "Sum Sample B: " + str( np.sum(clicks_sample_b) )

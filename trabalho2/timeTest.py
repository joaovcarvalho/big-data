import csv
import json
import numpy as np
import scipy.stats as stats
#import matplotlib.pyplot as plt

types_count = {}
DICT_KEY = " tempo"

def get_times(dataset):
    times = []
    for row in dataset:
        times.append( float(row[DICT_KEY]))
    return times

print "Population ==="
with open('populacao_tempo.csv', 'rb') as csvfile:
    dataset_population = csv.DictReader(csvfile, delimiter=';', quotechar='|')

    pop_times       = np.array(get_times(dataset_population));
    std_deviant_pop = np.std(pop_times)
    average_pop     = np.average(pop_times)

    print "STD: " + str(std_deviant_pop)
    print "AVG: " + str(average_pop)

print "Sample ==="
with open('amostra_tempo.csv', 'rb') as csvfile:
    dataset_sample = csv.DictReader(csvfile, delimiter=';', quotechar='|')

    sample_times       = np.array(get_times(dataset_sample));
    std_deviant_sample = np.std(sample_times)
    average_sample     = np.average(sample_times)

    print "STD: " + str(std_deviant_sample)
    print "AVG: " + str(average_sample)

z = (average_sample - average_pop) / (std_deviant_pop / np.sqrt(sample_times.size))
print "Z: " + str(z)
p_value = 1 - stats.norm.cdf(z)
print "P-value: " + str(p_value)

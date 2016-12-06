from __future__ import print_function
import pickle as pk
import numpy as np

dataset = np.genfromtxt('preprocessed_test.csv', delimiter=',')
p_file = open('model.pickle',"rb")
model = pk.load(p_file)

results = model.predict_proba(dataset)
# print results.shape
# print results[0]

f = open('results.txt', "wb")
for x in results:
    print(x[1], file=f)

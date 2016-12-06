import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

FILE_PATH = 'preprocessed_train.csv'

print "Loading dataset from file %s" % (FILE_PATH)
preprocessed_dataset = np.genfromtxt(FILE_PATH, delimiter=',',skip_header=1)
original_dataset     = np.genfromtxt('train_file.csv', delimiter=',', skip_header=1)
print "Datasets loaded..."



# print dataset[0]
# mean = dataset.mean(axis=0)
# std  = dataset.std(axis=0)

# bar_chart(mean)
# bar_chart(std)

# print "Columns: %d" % (dataset.shape[1])

def bar_chart(data):
    for i,n in enumerate(data):
        plt.bar( [i], [data[i]], width=0.35, label=n)
    plt.show()

def correlation_between_each_column_and_class(dataset):
    rho, p_value = stats.spearmanr(dataset)

    correlation_with_class = rho[:, -1]
    p_value_with_class = p_value[:, -1]
    plt.plot(correlation_with_class)
    plt.plot(p_value_with_class)
    plt.show()

# correlation_between_each_column_and_class(dataset)

def porcentage_of_each_class(dataset):
    dataset_zero = dataset[ dataset[:,-1] == 0.0]
    dataset_one = dataset[ dataset[:,-1] == 1.0]

    total_dataset =  dataset.shape[0]
    total_zero    = dataset_zero.shape[0] * 1.0
    total_one     = dataset_one.shape[0] * 1.0

    porcentage_zero = total_zero / total_dataset
    porcentage_one = total_one / total_dataset

    print "Porcentagem da classe 0: %f " % (porcentage_zero)
    print "Porcentagem da classe 1: %f " % (porcentage_one)

# porcentage_of_each_class(dataset)

def show_hist_for_each_column(dataset):
    dataset_zero = dataset[ dataset[:,-1] == 0.0]
    dataset_one = dataset[ dataset[:,-1] == 1.0]

    for index in range(dataset.shape[1]):
        column_zero = dataset_zero[:,index]
        column_one = dataset_one[:,index]

        plt.hist([column_zero, column_one], stacked=True, color=["r","b"])
        plt.show()

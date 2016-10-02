import csv
import json
import numpy as np
import scipy.stats as stats

def getDictFromCSV(file_path):
    with open(file_path, 'rb') as csvfile:
        dataset = csv.DictReader(csvfile, delimiter=',', quotechar='|')

        number_faces_dataset = []
        imdb_score_dataset   = []
        for row in dataset:
            try:
                number_faces_dataset.append( float(row["facenumber_in_poster"]))

                try:
                    score = float(row["imdb_score"])
                    if(score > 10 or score < 0):
                        raise ValueError

                    imdb_score_dataset.append( score )
                except ValueError:
                    number_faces_dataset.pop()

            except ValueError:
                pass

        return (number_faces_dataset, imdb_score_dataset)

number_faces, imdb_score = getDictFromCSV("movie_metadata.csv")
print number_faces
print imdb_score

rho, p_value = stats.spearmanr(number_faces, imdb_score, axis=None)
print "Rho: " + str(rho)
print "P-value: " + str(p_value)

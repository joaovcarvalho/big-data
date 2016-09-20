import csv
import json
import numpy as np
#import matplotlib.pyplot as plt

types_count = {}

with open('populacao_tempo.csv', 'rb') as csvfile:
    dataset_populacao = csv.DictReader(csvfile, delimiter=';', quotechar='|')

    tempos = []
    for row in dataset_populacao:
        tempos.append( float(row[" tempo"]))

    tempos = np.array(tempos)
    std_deviant_pop = np.std(tempos)
    average_pop     = np.average(tempos)

    print "STD: " + str(std_deviant_pop)
    print "AVG: " + str(average_pop)
    # # Counting the number of pokemons in each type
    # for row in pokemons:
    #     for type_i in range(1,2):
    #         pokemon_type = row["Type "+str(type_i)]

    #         if pokemon_type in types_count:
    #             types_count[pokemon_type] = types_count[pokemon_type] + 1
    #         else:
    #             types_count[pokemon_type] = 1

    # # Opening types strengths and weaknesses
    # with open('types.json', 'rb') as typesFile:
    #     types = json.loads(typesFile.read())

    # statistics = {}
    # for typeObject in types:
    #     typeName = typeObject['name']
    #     # print "Type " + typeName + ": " + str(types_count[typeName])
    #     countStrongAgainst = sum( map(lambda x: types_count[x], typeObject['strengths']) )
    #     countWeakAgainst   = sum( map(lambda x: types_count[x], typeObject['weaknesses']) )
    #     countImmunes       = sum( map(lambda x: types_count[x], typeObject['immunes']) )

    #     statistics[typeName] = {}
    #     statistics[typeName]['strong'] = countStrongAgainst
    #     statistics[typeName]['weak'] = countWeakAgainst
    #     statistics[typeName]['immune'] = countImmunes

    #     # print "Type " + typeName + " is strong against " + str(countStrongAgainst) + " pokemons."
    #     # print "Type " + typeName + " is weak against " + str(countWeakAgainst) + " pokemons."
    #     # print "Type " + typeName + " has not effect to " + str(countImmunes) + " pokemons. \n"

    # # Graph
    # ind = np.arange(len(types))
    # width = 0.35

    # data_strong   = map(lambda (k,v): v['strong'], statistics.iteritems())
    # data_weak     = map(lambda (k,v): v['weak'], statistics.iteritems())
    # data_immune   = map(lambda (k,v): v['immune'], statistics.iteritems())
    # names         = map(lambda (k,v): k, statistics.iteritems())
    # print data
    # fig,ax = plt.subplot()
    # rects  = ax.bar(ind, data, width, color='r')

    # import colorsys
    # N = len(names)
    # HSV_tuples = [(x*2.0/N, 0.5, 0.5) for x in range(N)]
    # RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    # for i,n in enumerate(names):
    #     color = RGB_tuples[i]
    #     plt.bar( [i], [data_strong[i]], width=0.35, label=n, color=color)
    #     plt.bar( [i + 0.35], [data_weak[i]], width=0.35, color=color)

    # ax  = plt.gca()

    # ax.set_ylabel('# of Pokemons')
    # ax.set_title('Strengths and Weakness by Pokemon Type')
    # ax.set_xticks(ind + 0.35)
    # ax.set_xticklabels(tuple(names))

    # fig.plot()
    # plt.axis([0, 6, 0, 20])
    # plt.show()

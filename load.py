from regions import *
import csv
import numpy as np

def get_data_from_file(filename):
    data_raw = []
    with open(filename) as datafile:
        datafile_reader = csv.reader(datafile, delimiter=",", quotechar="\"")
        for row in datafile_reader:
            data_raw.append(row)

    by_region = []
    for i in range(REGION_COUNT):
        by_region.append(np.array([]))
    total = np.array([])

    for i in range(4, len(data_raw[0])):
        column_by_region = np.zeros(8)

        for j in range(1, len(data_raw)):
            country = data_raw[j][1]
            if region_map.__contains__(country):
                region = region_map[country]
                if(data_raw[j][i] != ""):
                    column_by_region[region] += int(data_raw[j][i])
            else:
                print("could not find region for " + country)

        column_total = 0
        for i in range(REGION_COUNT):
            by_region[i] = np.append(by_region[i], column_by_region[i])

            column_total += column_by_region[i]

        total = np.append(total, column_total)

    return (by_region, total)
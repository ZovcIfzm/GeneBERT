# Script to convert labels into categories based on arguments

import argparse
import numpy as np
import pandas as pd



def convertLabels(trainFile, testFile, validFile, lower, upper):
    train, test, valid = None, None, None

    # Read in histone modification data from text files
    with open(trainFile, "r") as text_file:
        train_lines = text_file.read().splitlines()
        train = np.asarray([np.asarray(line.split(",")[1:])for line in train_lines])

    with open(testFile, "r") as text_file:
        test_lines = text_file.read().splitlines()
        test = np.asarray([np.asarray(line.split(",")[1:])for line in test_lines])

    with open(validFile, "r") as text_file:
        valid_lines = text_file.read().splitlines()
        valid = np.asarray([np.asarray(line.split(",")[1:])for line in valid_lines])

    #Concatenate data into single numpy matrix
    Cell1_concat = np.concatenate((train, test, valid), axis=0)

    #Convert numpy matrix to pandas dataframe
    df = pd.DataFrame(Cell1_concat)
    df = df.apply(pd.to_numeric)
    cols = ["hm1", "hm2", "hm3", "hm4", "hm5"]
    df.columns = cols


    df = df.applymap(lambda x: 'low' if x < lower else ('medium' if (x >= lower and x <= upper) else 'high'))

    print(df.head())


parser = argparse.ArgumentParser(description='CreateLabels')
parser.add_argument('--lower', type=int, default=-5, help='lower limit for middle category')
parser.add_argument('--upper', type=int, default=2, help='upper limit for middle category')
parser.add_argument('--trainFile', type=str, default='data/Cell1.train.csv', help='Path to train data')
parser.add_argument('--testFile', type=str, default='data/Cell1.test.csv', help='Path to test data')
parser.add_argument('--validFile', type=str, default='data/Cell1.valid.csv', help='Path to valid data')


args = parser.parse_args()

lower = args.lower
upper = args.upper

# Labels < lower -> 'low'
# Labels >= lower <= upper -> 'medium'
# Labels > upper -> 'high'


convertLabels(args.trainFile, args.testFile, args.validFile, args.lower, args.upper)




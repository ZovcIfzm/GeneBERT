# Script to convert labels into categories based on arguments

import argparse
import numpy as np
import pandas as pd


# Example command: python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file Cell1Conv.test.csv

def convertLabels(cell_file, output_file):
# From find_features()
    histones = None

    # Read in histone modification data from text files

    with open(cell_file, "r") as text_file:
        histones_lines = text_file.read().splitlines()
        histones = np.asarray([np.asarray(line.split(",")[1:]) for line in histones_lines])

    #Convert numpy matrix to pandas dataframe
    df = pd.DataFrame(histones)
    df = df.apply(pd.to_numeric)
    df.columns = ["hm1", "hm2", "hm3", "hm4", "hm5"]
    
    allowed_vals = [0,1]
    df[~df.isin(allowed_vals)] = "h"
    df = df.replace(0, 'l')
    df = df.replace(1, "m")

    def concat_row(row):
        return row["hm1"] + row["hm2"] + row["hm3"] + row["hm4"] + row["hm5"]

    df['concat'] = df.apply (lambda row: concat_row(row), axis=1)

    df['concat'].to_csv(output_file, index=False, header=False)

parser = argparse.ArgumentParser(description='CreateLabels')
parser.add_argument('--cell_file', type=str, default='data/Cell1.test.csv', help='Path to file for cell 1')
parser.add_argument('--output_file', type=str, default="output_cell.csv", help="path for output file")

args = parser.parse_args()

convertLabels(args.cell_file, args.output_file)




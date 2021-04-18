# Script to convert labels into categories based on arguments

import argparse
import numpy as np
import pandas as pd


# Example command: python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file Cell1Conv.test.csv
# python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file temporary_testing.tsv

'''
python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file C1Test.tsv
python3 convertHistoneLabels.py --cell_file data/Cell2.test.csv --output_file C2Test.tsv
python3 convertHistoneLabels.py --cell_file data/Cell1.train.csv --output_file C1Train.tsv
python3 convertHistoneLabels.py --cell_file data/Cell2.train.csv --output_file C2Train.tsv
python3 convertHistoneLabels.py --cell_file data/Cell1.valid.csv --output_file C1Valid.tsv
python3 convertHistoneLabels.py --cell_file data/Cell2.valid.csv --output_file C2Valid.tsv
'''


def convertLabels(cell_file, output_file):
# From find_features()
    histones = None

    # Read in histone modification data from text files

    with open(cell_file, "r") as text_file:
        histones_lines = text_file.read().splitlines()
        histones = np.asarray([np.asarray(line.split(",")) for line in histones_lines])

    #Convert numpy matrix to pandas dataframe
    full_df = pd.DataFrame(histones)
    full_df.columns = ["id", "hm1", "hm2", "hm3", "hm4", "hm5"]

    full_df["id"] = full_df["id"].str[:15]

    histone_df = pd.DataFrame({"hm1": full_df["hm1"], "hm2": full_df["hm2"], "hm3": full_df["hm3"], "hm4": full_df["hm4"], "hm5": full_df["hm5"]})
    histone_df = histone_df.apply(pd.to_numeric)

    allowed_vals = [0,1]
    histone_df[~histone_df.isin(allowed_vals)] = "h"
    histone_df = histone_df.replace(0, 'l')
    histone_df = histone_df.replace(1, "m")

    def concat_row(row):
        return row["hm1"] + row["hm2"] + row["hm3"] + row["hm4"] + row["hm5"]

    histone_df['concat'] = histone_df.apply (lambda row: concat_row(row), axis=1)

    new_df = pd.DataFrame({"id": full_df["id"], "token": histone_df["concat"]})

    # combines rows where the cell is the same, removes cell from dataframe
    # that way we just have sentences of histone modification tokens for each cell
    array_agg = lambda x: ' '.join(x.astype(str))
    grp_df = new_df.groupby(['id']).agg({'token': array_agg})

    grp_df.to_csv(output_file, sep='\t', index=False, header=False)

parser = argparse.ArgumentParser(description='CreateLabels')
parser.add_argument('--cell_file', type=str, default='data/Cell1.test.csv', help='Path to file for cell 1')
parser.add_argument('--output_file', type=str, default="output_cell.csv", help="path for output file")

args = parser.parse_args()

convertLabels(args.cell_file, args.output_file)




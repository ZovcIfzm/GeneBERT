# Script to convert labels into categories based on arguments

import argparse
import numpy as np
import pandas as pd
import csv

# Example command: python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file Cell1Conv.test.csv
# python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file temporary_testing.tsv

'''
python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file C1Test.csv
python3 convertHistoneLabels.py --cell_file data/Cell2.test.csv --output_file C2Test.csv
python3 convertHistoneLabels.py --cell_file data/Cell1.train.csv --output_file C1Train.csv
python3 convertHistoneLabels.py --cell_file data/Cell2.train.csv --output_file C2Train.csv
python3 convertHistoneLabels.py --cell_file data/Cell1.valid.csv --output_file C1Valid.csv
python3 convertHistoneLabels.py --cell_file data/Cell2.valid.csv --output_file C2Valid.csv

python3 transformHistoneData.py --cell1_file data/Cell1.train.csv --cell2_file data/Cell2.train.csv --output_file DiffTrain.csv
python3 transformHistoneData.py --cell1_file data/Cell1.test.csv --cell2_file data/Cell2.test.csv --output_file DiffTest.csv
python3 transformHistoneData.py --cell1_file data/Cell1.valid.csv --cell2_file data/Cell2.valid.csv --output_file DiffValid.csv
'''


def concat_row(row):
    return row["hm1"] + row["hm2"] + row["hm3"] + row["hm4"] + row["hm5"]

def transformSingleCellHistones(cell_file, output_file):
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

    grp_df.to_csv(output_file, index=False, header=False, quoting=csv.QUOTE_NONE)

def transformDualCellHistones(cell1_file, cell2_file, output_file):
    cell1, cell2 = None, None

    # Read in histone modification data from text files
    with open(cell1_file, "r") as text_file:
        cell1_lines = text_file.read().splitlines()
        cell1 = np.asarray([np.asarray(line.split(",")) for line in cell1_lines])

    with open(cell2_file, "r") as text_file:
        cell2_lines = text_file.read().splitlines()
        cell2 = np.asarray([np.asarray(line.split(",")) for line in cell2_lines])

    #Convert numpy matrix to pandas dataframe
    cell1_df = pd.DataFrame(cell1)
    cell2_df = pd.DataFrame(cell2)

    cell1_df.columns = ["id", "hm1", "hm2", "hm3", "hm4", "hm5"]
    cell2_df.columns = ["id", "hm1", "hm2", "hm3", "hm4", "hm5"]

    cell1_df["id"] = cell1_df["id"].str[:15]
    cell2_df["id"] = cell2_df["id"].str[:15]

    cell1_hm_df = pd.DataFrame({"hm1": cell1_df["hm1"], "hm2": cell1_df["hm2"], "hm3": cell1_df["hm3"], "hm4": cell1_df["hm4"], "hm5": cell1_df["hm5"]})
    cell2_hm_df = pd.DataFrame({"hm1": cell2_df["hm1"], "hm2": cell2_df["hm2"], "hm3": cell2_df["hm3"], "hm4": cell2_df["hm4"], "hm5": cell2_df["hm5"]})
    
    cell1_hm_df = cell1_hm_df.apply(pd.to_numeric)
    cell2_hm_df = cell2_hm_df.apply(pd.to_numeric)

    diff_cell = pd.DataFrame({"hm1": cell1_hm_df["hm1"] - cell2_hm_df["hm1"], "hm2": cell1_hm_df["hm2"] - cell2_hm_df["hm2"], "hm3": cell1_hm_df["hm3"] - cell2_hm_df["hm3"], "hm4": cell1_hm_df["hm4"] - cell2_hm_df["hm4"], "hm5": cell1_hm_df["hm5"] - cell2_hm_df["hm5"]})
    diff_cell = diff_cell.applymap(lambda x: 'l' if x < 0 else ('m' if (x >= 0 and x < 1) else 'h'))
    diff_cell['concat'] = diff_cell.apply(lambda row: concat_row(row), axis=1)

    new_df = pd.DataFrame({"id": cell1_df["id"], "token": diff_cell["concat"]})

    # combines rows where the cell is the same, removes cell from dataframe
    # that way we just have sentences of histone modification tokens for each cell
    array_agg = lambda x: ' '.join(x.astype(str))
    grp_df = new_df.groupby(['id']).agg({'token': array_agg})
    grp_df.columns=["id"]
    mask_df = pd.DataFrame({"id": grp_df["id"] + " <mask>"})

    mask_df.to_csv(output_file, index=False, header=False, quoting=csv.QUOTE_NONE)

parser = argparse.ArgumentParser(description='CreateLabels')
parser.add_argument('--cell1_file', type=str, default='data/Cell1.test.csv', help='Path to file for cell 1')
parser.add_argument('--cell2_file', type=str, default='data/Cell1.test.csv', help='Path to file for cell 2')
parser.add_argument('--output_file', type=str, default="output_cell.csv", help="path for output file")

args = parser.parse_args()

# transformSingleCellHistones(args.cell1_file, args.output_file)
transformDualCellHistones(args.cell1_file, args.cell2_file, args.output_file)
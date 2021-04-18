# Script to convert labels into categories based on arguments

import argparse
import numpy as np
import pandas as pd


# Example command: python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file Cell1Conv.test.csv
# python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file temporary_testing.tsv

'''
python3 convertHistoneLabels.py --cell_file data/Cell1.test.csv --output_file C1Test.csv
python3 convertHistoneLabels.py --cell_file data/Cell2.test.csv --output_file C2Test.csv
python3 convertHistoneLabels.py --cell_file data/Cell1.train.csv --output_file C1Train.csv
python3 convertHistoneLabels.py --cell_file data/Cell2.train.csv --output_file C2Train.csv
python3 convertHistoneLabels.py --cell_file data/Cell1.valid.csv --output_file C1Valid.csv
python3 convertHistoneLabels.py --cell_file data/Cell2.valid.csv --output_file C2Valid.csv

python3 createTSVData.py --cell1_file word_tokenized/C1Test.csv --cell2_file word_tokenized/C2Test.csv --diff_file C12TestDiff.tsv --output_file Test.tsv
python3 createTSVData.py --cell1_file word_tokenized/C1Train.csv --cell2_file word_tokenized/C2Train.csv --diff_file C12TrainDiff.tsv --output_file Train.tsv
python3 createTSVData.py --cell1_file word_tokenized/C1Valid.csv --cell2_file word_tokenized/C2Valid.csv --diff_file C12ValidDiff.tsv --output_file Valid.tsv
'''


def createTSVData(cell1_file, cell2_file, diff_file, output_file):
# From find_features()
    cell1, cell2 = None, None

    # read in tokens from cell 1 and cell 2 files 
    with open(cell1_file, "r") as text_file:
        cell1_lines = text_file.read().splitlines()

    with open(cell2_file, "r") as text_file:
        cell2_lines = text_file.read().splitlines()

    with open(diff_file, "r") as text_file:
        diff_lines = text_file.read().splitlines()

    combined = np.asarray([cell1_lines[i].strip('"') + '\t' + cell2_lines[i].strip('"') + '\t' + diff_lines[i].strip('"') for i in range(len(cell1_lines))])
    df = pd.DataFrame(combined)

    df.to_csv(output_file, index=False, header=False)

parser = argparse.ArgumentParser(description='CreateLabels')
parser.add_argument('--cell1_file', type=str, default='', help='Path to file for cell 1')
parser.add_argument('--cell2_file', type=str, default='', help='Path to file for cell 1')
parser.add_argument('--diff_file', type=str, default='', help='')
parser.add_argument('--output_file', type=str, default="output_cell.csv", help="path for output file")

args = parser.parse_args()

createTSVData(args.cell1_file, args.cell2_file, args.diff_file, args.output_file)




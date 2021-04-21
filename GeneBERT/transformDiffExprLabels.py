# Script to convert labels into categories based on arguments

import argparse
import numpy as np
import pandas as pd


'''
python3 convertLabelsRegr.py --c1_file data/Cell1.expr.csv --c2_file data/Cell2.expr.csv --ref_file data/Cell1.test.csv --output_file C12TestDiff.tsv
python3 convertLabelsRegr.py --c1_file data/Cell1.expr.csv --c2_file data/Cell2.expr.csv --ref_file data/Cell1.train.csv --output_file C12TrainDiff.tsv
python3 convertLabelsRegr.py --c1_file data/Cell1.expr.csv --c2_file data/Cell2.expr.csv --ref_file data/Cell1.valid.csv --output_file C12ValidDiff.tsv
'''

def convertLabelsRegr(c1_file, c2_file, ref_file, output_file):

	cell1_df, cell2_df, ref = None, None, None

	with open(c1_file, "r") as text_file:
		cell1_expr_lines = text_file.read().splitlines()
		cell1_expr = np.asarray([np.asarray(line.split(",")) for line in cell1_expr_lines])

	with open(c2_file, "r") as text_file:
		cell2_expr_lines = text_file.read().splitlines()
		cell2_expr = np.asarray([np.asarray(line.split(","))for line in cell2_expr_lines])

	with open(ref_file, "r") as text_file:
		ref_lines = text_file.read().splitlines()
		ref = [line.split(",")[0][:15] for line in ref_lines]

	cell1_df = pd.DataFrame(cell1_expr)
	cell2_df = pd.DataFrame(cell2_expr)

	cell1_df.columns = ["id", "expr"]
	cell2_df.columns = ["id", "expr"]


	cell1_df = cell1_df[cell1_df["id"].isin(ref)]
	cell2_df = cell2_df[cell2_df["id"].isin(ref)]


	cell1_df["expr"] = cell1_df["expr"].apply(pd.to_numeric)
	cell2_df["expr"] = cell2_df["expr"].apply(pd.to_numeric)

	cell1_df["expr"] = np.log(cell1_df["expr"]+1)
	cell2_df["expr"] = np.log(cell2_df["expr"]+1)


	new_df = pd.DataFrame({"expr": cell1_df["expr"] - cell2_df["expr"]})
	new_df.to_csv(output_file, sep='\t', index=False, header=False)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='CreateLabels')
	parser.add_argument('--c1_file', type=str, required=True, help='Path to file for cell 1')
	parser.add_argument('--c2_file', type=str, required=True, help='Path to file for cell 2')
	parser.add_argument('--ref_file', type=str, required=True, help='Path to file for cell 2')
	parser.add_argument('--output_file', type=str, required=True, help='Path to output file')


	args = parser.parse_args()


	convertLabelsRegr(args.c1_file, args.c2_file, args.ref_file, args.output_file)




	# Test BERT with the categorical labels and compare metrics





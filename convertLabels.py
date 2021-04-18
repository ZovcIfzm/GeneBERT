# Script to convert labels into categories based on arguments

import argparse
import numpy as np
import pandas as pd



def convertLabels(c1_file, c2_file, output_file, lower, upper):

	cell1_df, cell2_df = None, None

	with open(c1_file, "r") as text_file:
		cell1_expr_lines = text_file.read().splitlines()
		cell1_expr = np.asarray([np.asarray(line.split(",")[1:]) for line in cell1_expr_lines])

	with open(c2_file, "r") as text_file:

		cell2_expr_lines = text_file.read().splitlines()
		cell2_expr = np.asarray([np.asarray(line.split(",")[1:])for line in cell2_expr_lines])


	cell1_df = pd.DataFrame(cell1_expr)
	cell2_df = pd.DataFrame(cell2_expr)

	cell1_df = cell1_df.apply(pd.to_numeric)
	cell2_df = cell2_df.apply(pd.to_numeric)

	cell1_df.columns = ["expr"]
	cell2_df.columns = ["expr"]

	cell1_df["expr"] = np.log(cell1_df["expr"]+1)
	cell2_df["expr"] = np.log(cell2_df["expr"]+1)

	new_df = cell1_df-cell2_df

	new_df = new_df.applymap(lambda x: 'l' if x < lower else ('m' if (x >= lower and x <= upper) else 'h'))

	new_df.to_csv(output_file, sep='\t', index=False, header=False)



# python3 convertLabels.py --lower -3 --upper 2 --c1_file data/Cell1.expr.csv --c2_file data/Cell2.expr.csv --output_file C12Diff.tsv

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='CreateLabels')
	parser.add_argument('--lower', type=int, default=-3, help='lower limit for middle category')
	parser.add_argument('--upper', type=int, default=2, help='upper limit for middle category')
	parser.add_argument('--c1_file', type=str, default='data/Cell1.expr.csv', help='Path to file for cell 1')
	parser.add_argument('--c2_file', type=str, default='data/Cell2.expr.csv', help='Path to file for cell 2')
	parser.add_argument('--output_file', type=str, default='tokenized_data/Cell1_Cell2_Conv.expr.csv', help='Path to output file')


	args = parser.parse_args()

	lower = args.lower
	upper = args.upper

	# Labels < lower -> 'low'
	# Labels >= lower <= upper -> 'medium'
	# Labels > upper -> 'high'


	convertLabels(args.c1_file, args.c2_file, args.output_file, args.lower, args.upper)




	# Test BERT with the categorical labels and compare metrics





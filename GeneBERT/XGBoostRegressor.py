from xgboost import XGBRegressor
import pandas as pd

import argparse
from sklearn import metrics

import scipy
import pickle
from pathlib import Path
import numpy as np

'''
python3 XGBoostRegressor.py --dataset_type raw_c > raw_c_results.txt
python3 XGBoostRegressor.py --dataset_type raw_d > raw_d_results.txt
python3 XGBoostRegressor.py --dataset_type raw > raw_results.txt
'''

# extract type of feature data from arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, required=True, help='type of data')
args = parser.parse_args()

# read base data
X_train, y_train, X_test, y_test = None, None, None, None
Xc_train, Xc_test = None, None
Xd_train, Xd_test = None, None

# Get labels (same across all datasets)
test_diff_df = pd.read_csv("diff_expr/C12TestDiff.tsv", delimiter='\t', header=None)
train_diff_df = pd.read_csv("diff_expr/C12TrainDiff.tsv", delimiter='\t', header=None)
valid_diff_df = pd.read_csv("diff_expr/C12ValidDiff.tsv", delimiter='\t', header=None)

y_test = test_diff_df
y_train = pd.concat([train_diff_df, valid_diff_df])

# Get features
if args.dataset_type == "raw_c" or args.dataset_type == "raw":
    c1_test_df = pd.read_csv("embeddings/C1TestEmbeddings.csv", header=None)
    c1_train_df = pd.read_csv("embeddings/C1TrainEmbeddings.csv", header=None)
    c1_valid_df = pd.read_csv("embeddings/C1ValidEmbeddings.csv", header=None)

    c2_test_df = pd.read_csv("embeddings/C2TestEmbeddings.csv", header=None)
    c2_train_df = pd.read_csv("embeddings/C2TrainEmbeddings.csv", header=None)
    c2_valid_df = pd.read_csv("embeddings/C2ValidEmbeddings.csv", header=None)

    c12_train_df = pd.concat([c1_train_df, c2_train_df], axis=1)
    c12_valid_df = pd.concat([c1_valid_df, c2_valid_df], axis=1)

    # Reset index prevents errors from combining Xd_train and Xc_train.
    # Unless you specify drop=True, it creates a new column that stores indices
    # We don't want this, because it essentially adds another feature that's just id
    Xc_train = pd.concat([c12_train_df, c12_valid_df]).reset_index(drop=True)
    Xc_test = pd.concat([c1_test_df, c2_test_df], axis=1)

if args.dataset_type == "raw_d" or args.dataset_type == "raw":
    diff_train = pd.read_csv("embeddings/DiffTrainEmbeddings.csv", header=None)
    diff_valid = pd.read_csv("embeddings/DiffValidEmbeddings.csv", header=None)

    # Reset index prevents errors from combining Xd_train and Xc_train.
    # Unless you specify drop=True, it creates a new column that stores indices
    # We don't want this, because it essentially adds another feature that's just id
    Xd_train = pd.concat([diff_train, diff_valid]).reset_index(drop=True)
    Xd_test = pd.read_csv("embeddings/DiffTestEmbeddings.csv", header=None)

if args.dataset_type == "raw_c":
    X_train = Xc_train
    X_test = Xc_test

if args.dataset_type == "raw_d":
    X_train = Xd_train
    X_test = Xd_test

if args.dataset_type == "raw":
    # print("len", len(Xc_train.columns), len(Xd_train.columns))
    X_train = pd.concat([Xc_train, Xd_train], axis=1)
    X_test = pd.concat([Xc_test, Xd_test], axis=1)

# If model is saved, load model, else train model
file_path = Path("saved_models/xgbr_"+args.dataset_type+".sav")
model = None
if file_path.is_file():
    model = pickle.load(open(file_path, 'rb'))
else:
    # print(X_train)
    X_train.columns = [i for i in range(len(X_train.columns))]
    y_train.columns = [i for i in range(len(y_train.columns))]
    X_test.columns = [i for i in range(len(X_test.columns))]
    y_test.columns = [i for i in range(len(y_test.columns))]

    model = XGBRegressor(random_state=0)
    model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("XGBoost regressor score: ", score)

ypred = model.predict(X_test)

R,p=scipy.stats.pearsonr(np.squeeze(y_test),np.squeeze(ypred))
MSE=metrics.mean_squared_error(y_test, ypred)

print("R:", R)
print("p:", p)
print("MSE:", MSE)
# Save model
pickle.dump(model, open(file_path, 'wb'))

# hyper parameter tuning, to work with later
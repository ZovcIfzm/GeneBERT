from xgboost import XGBRegressor
import pandas as pd

from sklearn import metrics

import scipy
import pickle
from pathlib import Path
import numpy as np

test_diff_df = pd.read_csv("C12TestDiff.tsv", delimiter='\t', header=None)
train_diff_df = pd.read_csv("C12TrainDiff.tsv", delimiter='\t', header=None)
valid_diff_df = pd.read_csv("C12ValidDiff.tsv", delimiter='\t', header=None)

c1_test_df = pd.read_csv("C1TestEmbeddings.csv", header=None)
c1_train_df = pd.read_csv("C1TrainEmbeddings.csv", header=None)
c1_valid_df = pd.read_csv("C1ValidEmbeddings.csv", header=None)

c2_test_df = pd.read_csv("C2TestEmbeddings.csv", header=None)
c2_train_df = pd.read_csv("C2TrainEmbeddings.csv", header=None)
c2_valid_df = pd.read_csv("C2ValidEmbeddings.csv", header=None)

# If model is saved, load model, else train model
file_path = Path("saved_models/xgbr.sav")
model = None
if file_path.is_file():
    model = pickle.load(open(file_path, 'rb'))
else:
    c12_train_df = pd.concat([c1_train_df, c2_train_df], axis=1)
    c12_valid_df = pd.concat([c1_valid_df, c2_valid_df], axis=1)
    X_train = pd.concat([c12_train_df, c12_valid_df])
    y_train = pd.concat([train_diff_df, valid_diff_df])

    print(X_train.head())
    X_test = pd.concat([c1_test_df, c2_test_df], axis=1)
    y_test = test_diff_df

    X_train.columns = [i for i in range(len(X_train.columns))]
    y_train.columns = [i for i in range(len(y_train.columns))]
    X_test.columns = [i for i in range(len(X_test.columns))]
    y_test.columns = [i for i in range(len(y_test.columns))]

    model = XGBRegressor(random_state=0)
    model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("XGBoost regressor score: ", score)

ypred = model.predict(X_test)

R2,p=scipy.stats.pearsonr(np.squeeze(y_test),np.squeeze(ypred))
MSE=metrics.mean_squared_error(y_test, ypred)

print(R2, p)
print(MSE)
# Save model
pickle.dump(model, open(file_path, 'wb'))

# hyper parameter tuning, to work with later

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_features():
    train, test, valid = None, None, None

    # Read in histone modification data from text files
    with open("data/Cell1.train.csv", "r") as text_file:
        train_lines = text_file.read().splitlines()
        train = np.asarray([np.asarray(line.split(",")[1:])for line in train_lines])

    with open("data/Cell1.test.csv", "r") as text_file:
        test_lines = text_file.read().splitlines()
        test = np.asarray([np.asarray(line.split(",")[1:])for line in test_lines])

    with open("data/Cell1.valid.csv", "r") as text_file:
        valid_lines = text_file.read().splitlines()
        valid = np.asarray([np.asarray(line.split(",")[1:])for line in valid_lines])

    #Concatenate data into single numpy matrix
    Cell1_concat = np.concatenate((train, test, valid), axis=0)

    #Convert numpy matrix to pandas dataframe
    df = pd.DataFrame(Cell1_concat)
    df = df.apply(pd.to_numeric)
    df.columns = ["hm1", "hm2", "hm3", "hm4", "hm5"]
        
    df.hist(bins=range(30), density=True)
    plt.suptitle("Density of histone modification values")
    plt.show()

def find_expression():
    cell1_df, cell2_df = None, None

    with open("data/Cell1.expr.csv", "r") as text_file:
        cell1_expr_lines = text_file.read().splitlines()
        cell1_expr = np.asarray([np.asarray(line.split(",")[1:]) for line in cell1_expr_lines])

    with open("data/Cell2.expr.csv", "r") as text_file:
        cell2_expr_lines = text_file.read().splitlines()
        cell2_expr = np.asarray([np.asarray(line.split(",")[1:])for line in cell2_expr_lines])

    #Convert numpy matrix to pandas dataframe
    cell1_df = pd.DataFrame(cell1_expr)
    cell2_df = pd.DataFrame(cell2_expr)

    cell1_df = cell1_df.apply(pd.to_numeric)
    cell2_df = cell2_df.apply(pd.to_numeric)

    cell1_df.columns = ["expr"]
    cell2_df.columns = ["expr"]

    cell1_df["expr"] = np.log(cell1_df["expr"]+1)
    cell2_df["expr"] = np.log(cell2_df["expr"]+1)

    new_df = cell1_df-cell2_df
        
    new_df.hist()
    plt.suptitle("log(cell1 expression + 1) - log(cell2 expression + 1)")
    plt.show()

find_expression()
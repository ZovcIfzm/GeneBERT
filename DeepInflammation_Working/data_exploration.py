
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
    return df
        
def show_features():
    df = find_features()
    df.hist(bins=range(30), density=True)
    plt.suptitle("Density of histone modification values")
    plt.show()

def find_feature_diff():
    train1, test1, valid1, train2, test2, valid2 = None, None, None, None, None, None

    # Read in histone modification data from text files
    with open("data/Cell1.train.csv", "r") as text_file:
        train1_lines = text_file.read().splitlines()
        train1 = np.asarray([np.asarray(line.split(",")[1:])for line in train1_lines])

    with open("data/Cell1.test.csv", "r") as text_file:
        test1_lines = text_file.read().splitlines()
        test1 = np.asarray([np.asarray(line.split(",")[1:])for line in test1_lines])

    with open("data/Cell1.valid.csv", "r") as text_file:
        valid1_lines = text_file.read().splitlines()
        valid1 = np.asarray([np.asarray(line.split(",")[1:])for line in valid1_lines])

    with open("data/Cell2.train.csv", "r") as text_file:
        train2_lines = text_file.read().splitlines()
        train2 = np.asarray([np.asarray(line.split(",")[1:])for line in train2_lines])

    with open("data/Cell2.test.csv", "r") as text_file:
        test2_lines = text_file.read().splitlines()
        test2 = np.asarray([np.asarray(line.split(",")[1:])for line in test2_lines])

    with open("data/Cell2.valid.csv", "r") as text_file:
        valid2_lines = text_file.read().splitlines()
        valid2 = np.asarray([np.asarray(line.split(",")[1:])for line in valid2_lines])

    #Concatenate data into single numpy matrix
    Cell1_concat = np.concatenate((train1, test1, valid1), axis=0)
    Cell2_concat = np.concatenate((train2, test2, valid2), axis=0)

    #Convert numpy matrix to pandas dataframe
    cell1_df = pd.DataFrame(Cell1_concat)
    cell2_df = pd.DataFrame(Cell2_concat)

    cell1_df = cell1_df.apply(pd.to_numeric)
    cell2_df = cell2_df.apply(pd.to_numeric)

    cell1_df.columns = ["hm1", "hm2", "hm3", "hm4", "hm5"]
    cell2_df.columns = ["hm1", "hm2", "hm3", "hm4", "hm5"]

    df = pd.DataFrame({"hm1": cell1_df["hm1"] - cell2_df["hm1"], "hm2": cell1_df["hm2"] - cell2_df["hm2"], "hm3": cell1_df["hm3"] - cell2_df["hm3"], "hm4": cell1_df["hm4"] - cell2_df["hm4"], "hm5": cell1_df["hm5"] - cell2_df["hm5"]})
    #print(df["hm1"])
    #print(cell1_df["hm1"])
    #print(cell2_df["hm1"])
    
    return df

def show_feature_diff():
    df = find_feature_diff() 
    df.hist(bins=[i for i in range(-15, 15)], density=True)
    plt.suptitle("Density of difference in histone modification values")
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

# Finding ranges
#Helper functions
def find_thresholds(hm_list):
    print(hm_list[int(len(hm_list)/3.0)], hm_list[int(2*len(hm_list)/3.0)])

def find_percentiles(hm_list):
    print(hm_list.index(1)/float(len(hm_list)), hm_list.index(2)/float(len(hm_list)))

#Find ranges for single histone file 
def find_ranges():
    df = find_features()

    hm1 = df["hm1"].tolist()
    hm2 = df["hm2"].tolist()
    hm3 = df["hm3"].tolist()
    hm4 = df["hm4"].tolist()
    hm5 = df["hm5"].tolist()

    hm1.sort()
    hm2.sort()
    hm3.sort()
    hm4.sort()
    hm5.sort()
    
    find_percentiles(hm1)
    find_percentiles(hm2)
    find_percentiles(hm3)
    find_percentiles(hm4)
    find_percentiles(hm5)

#Find ranges for difference between two histone files
def find_histone_diff_ranges():
    df = find_feature_diff()

    hm1 = df["hm1"].tolist()
    hm2 = df["hm2"].tolist()
    hm3 = df["hm3"].tolist()
    hm4 = df["hm4"].tolist()
    hm5 = df["hm5"].tolist()

    hm1.sort()
    hm2.sort()
    hm3.sort()
    hm4.sort()
    hm5.sort()

    def find_thresholds(hm_list):
        print(hm_list[int(len(hm_list)/3.0)], hm_list[int(2*len(hm_list)/3.0)])

    def find_percentiles(hm_list):
        print(hm_list.index(0)/float(len(hm_list)), hm_list.index(1)/float(len(hm_list)))
    
    find_percentiles(hm1)
    find_percentiles(hm2)
    find_percentiles(hm3)
    find_percentiles(hm4)
    find_percentiles(hm5)

find_histone_diff_ranges()
show_feature_diff()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

print(df["hm1"].values)

df.hist(bins=range(30), density=True, xlabelsize=10)
plt.suptitle("Density of histone modification values")
plt.show()

'''

df_show = df["hm1"]
df_show.hist()
plt.show()
'''
'''

plt.hist(df["hm1"].values, bins= int(180/5), color="blue", edgecolor= 'black')
plt.title("HM 1 value histogram")
plt.xlabel("values")
plt.ylabel("population")
plt.show()
'''
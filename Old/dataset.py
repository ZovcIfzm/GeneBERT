import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class DeepInflammationDataset(Dataset):

    def __init__(self, c1_data, c2_data):
        """
        - c1_data: pandas dataframe of the cell 1 data
        - c2_data: pandas dataframe of the cell 2 data
        """
        super().__init__()
        
        self.c1_data = c1_data
        self.c2_data = c2_data
    
    def __len__(self):
        """Return the number of samples."""

        return self.c1_data.shape[0] + self.c2_data.shape[0]

    def __getitem__(self, idx):
        """
        Input:
            - idx: index of sample to retrieve
        Return:
        	- sample: the idx'th sample

        	- label: the label for the sample
        """

        c = pd.concat([self.c1_data['expr'], self.c2_data['expr']], axis=0, ignore_index=True)

        return (c.iloc[idx])



if __name__ == '__main__':

	c1_input_train = pd.read_csv('tokenized_data/Cell1Conv.train.csv', names=['expr'])
	#c1_input_test = pd.read_csv('tokenized_data/Cell1Conv.test.csv')
	c1_input_valid = pd.read_csv('tokenized_data/Cell1Conv.valid.csv', names=['expr'])

	c2_input_train = pd.read_csv('tokenized_data/Cell2Conv.train.csv', names=['expr'])
	#c2_input_test = pd.read_csv('tokenized_data/Cell2Conv.test.csv')
	c2_input_valid = pd.read_csv('tokenized_data/Cell2Conv.valid.csv', names=['expr'])


	train_dataset = DeepInflammationDataset(c1_input_train, c2_input_train)


	for i in range(len(train_dataset)):
		sample = train_dataset[i]

		print(sample)

		break

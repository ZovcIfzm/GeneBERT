# EECS498/598 HW4
import os
import pickle
from collections import Counter

import torch
from torchtext.vocab import Vocab
from torch.utils.data import Dataset
import pandas as pd
from nltk import word_tokenize

import numpy as np
import csv

def find_missing_dialogue_id(df):
    """
    Check if there are missing Dialogue_ID in df. If yes, return missing Dialogue_ID.
    Input:
        - df: pandas dataframe in which we need to find missing Dialogue_ID.
    Return:
        - ids: list of missing Dialogue_ID.
        return [] if there is no missing Dialogue_ID
    
    Hint: you can check every Dialogue_ID in the range [0, max(Dialogue_ID)], if
    it is not in df, add it to the output list.
    """

    max_val = max(df["Dialogue_ID"])
    #res = df["Dialogue_ID"].isin(range(0,max_val))
    #print(res)   
    
    results = []

    for i in range(0,max_val):
      if i not in df["Dialogue_ID"].values:
        results.append(i)
    
    return results


def get_class_stats(train_data, label_index):
    """
    Calculate the number of utterances for each emotion label.
    Input:
        - train_data: training data on which to get statistics.
        - label_index: indices for labels
    Return:
        - output: a list of length len(label_index) (in this case 7), that stores
            the number of utterances for each emotion label. Results should be
            in the order specified by label_index.
    """

    output = [0,0,0,0,0,0,0]

    for label,value in label_index.items():
      count = train_data["Emotion"].loc[train_data["Emotion"] == label].count()
      output[value] = count
    
    return output


def get_vocabulary(train_file, min_freq, specials=['<unk>']):
    """
    Preprocess utterances in training data and create vocabulary.
    Read train_file with pd.read_csv().
    Input:
        - train_file: filename of training data.
        - min_freq: the minimum frequency needed to inlcude a token in the
            vocabulary. e.g. set min_freq = 3 to only include tokens that
            appear at least 3 times in the training data.
        - specials: list of special tokens to add in the vocabulary.
    Return:
        - vocab: torchtext.vocab.Vocab instance that stores the vocabulary of
            training data.
    Preprocess steps:
        1. convert utterance to lower case
        2. tokenize utterance with nltk.word_tokenize
        3. update the counter
        4. create vocabulary using counter
    """

    df = pd.read_csv(train_file)
    df["Utterance"] = df["Utterance"].str.lower()
    counter = Counter()

    for line in df["Utterance"].values:
      counter.update(word_tokenize(line))

    vocab = Vocab(counter, min_freq=min_freq, specials=specials)

    return vocab


def load_glove_matrix(gloveFile, vocab):
    """
    Load the pretrained glove embedding, only keep tokens that are in our vocabulary.
    Input:
        - gloveFile: file that stores the pretrained glove embedding.
        - vocab: torchtext.vocab.Vocab instance that stores the vocabulary of the
        training data.
    Return:
        - W: torch tensor with shape (num_vocab, 300), where num_tokens is the
            size of vocabulary in the training data (with special tokens <unk>).
            Each row of W is the Glove embedding of a token, e.g. W[i] is the
            embedding for the token at index i.
        Note: use all zeros as the embedding for <unk>
    
    Note: if a token in the vocabulary does not appear in the GloVe file, its
        embedding should be zeros.
    Hint: you can use torch.zeros() to create the tensor first, and assign values
        for some rows.
    """
    f = open(gloveFile,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        if word in vocab.itos:
          wordEmbedding = np.array([float(value) for value in splitLines[1:]])
          gloveModel[word] = wordEmbedding

    zeros_array = np.zeros(300)
    np_matrix = np.asarray([np.asarray(gloveModel[word]) if word in gloveModel else zeros_array for word in vocab.itos])

    W = torch.from_numpy(np_matrix).float()
    return W


class MELDDataset(Dataset):
    """Dataset for MELD."""

    def __init__(self, data, vocab, label_index, audio_emb, W):
        """
        - data: pandas dataframe of the data.
        - vocab: torchtext.vocab.Vocab instance that stores the vocabulary of
        training data.
        - label_index: mapping of each label to its assigned index.
        - audio_emb: dictionary of audio embedding.
        - W: W returned from load_glove_matrix().
        """

        self.data = data
        self.vocab = vocab
        self.label_index = label_index
        self.audio_emb = audio_emb
        self.W = W

        super().__init__()
    
    def __len__(self):
        """Return the number of dialogues."""
        return self.data["Dialogue_ID"].max()

    def __getitem__(self, idx):
        """
        Input:
            - idx: dialogue_id
        Return:
            - text_emb: list of torch tensors with shape (num_tokens, 300) that
                represents the text embedding of each utterance in dialogue_id,
                num_tokens is the number of tokens in that utterance.
                The length of text_emb should be the number of utterances in dialogue_id.
                Remember to take the same preprocessing steps as before.
            - audio_emb: torch tensor with shape (num_utterance, 1611),
                where num_utterance is the number of utterances in dialogue_id,
                1611 is the number of features for audio embedding of each utterance.
            - label: torch tensor with shape (num_utterance,) that stores the
                label index for each utterance.
        """
        dialogue_rows = self.data.loc[self.data["Dialogue_ID"] == idx]
        text_emb = [torch.stack([self.W[self.vocab.stoi[token]] for token in word_tokenize(utterance.lower())]) for utterance in dialogue_rows["Utterance"]]
        audio_emb = torch.stack([self.audio_emb[str(idx) +"_"+ str(utterance_id)] for utterance_id in dialogue_rows["Utterance_ID"]])

        label = torch.tensor([self.label_index[emotion] for emotion in dialogue_rows["Emotion"]])
        return text_emb, audio_emb, label

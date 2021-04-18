import sys
sys.path.append(GOOGLE_DRIVE_PATH)

import pickle

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch

from dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

test = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH, 'MELD', 'test_sent_emo.csv'))
dev = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH,'MELD', 'dev_sent_emo.csv'))
train = pd.read_csv(os.path.join(GOOGLE_DRIVE_PATH, 'MELD', 'train_sent_emo.csv'))

print(train.head())

label_index = {
    'neutral': 0, 'surprise': 1, 'fear': 2,
    'sadness': 3, 'joy':4, 'disgust': 5,
    'anger': 6
}

class_stats = torch.tensor(get_class_stats(train, label_index))

import nltk
nltk.download('punkt')

vocab = get_vocabulary(os.path.join(GOOGLE_DRIVE_PATH, 'MELD', 'train_sent_emo.csv'), 1)
print(len(vocab.freqs))
print(len(vocab.stoi))
print(len(vocab.itos))

W = load_glove_matrix(os.path.join(GOOGLE_DRIVE_PATH, 'glove.6B.300d.txt'), vocab)
print(torch.equal(W[0], torch.zeros(300))) # all zeros for <unk>
print(W.shape)
print(W[1, :5])

from utils import meld_collate_fn

trn_data = MELDDataset(train, vocab, label_index,
                      train_audio_emb, W)
trn_loader = DataLoader(trn_data, 4, shuffle=False, collate_fn=meld_collate_fn)
text, audio, label, num_utt = next(iter(trn_loader))
print(text.shape)
print(text[0, :10, 0])
print(audio.shape)
print(label)
print(num_utt)

trn_data = MELDDataset(train, vocab, label_index, train_audio_emb, W)
val_data = MELDDataset(dev, vocab, label_index, val_audio_emb, W)
test_data = MELDDataset(test, vocab, label_index, test_audio_emb, W)
# Create dataloaders
trn_loader = DataLoader(trn_data, 32, shuffle=True, collate_fn=meld_collate_fn)
val_loader = DataLoader(val_data, 32, shuffle=True, collate_fn=meld_collate_fn)
test_loader = DataLoader(test_data, 32, shuffle=True, collate_fn=meld_collate_fn)



# Model

import random
from model import TextCNNEncoder

# parameters for network structure
kernel_size = [3, 4, 5]
num_channel = 100
# set random seed
random.seed(0)
torch.manual_seed(0)

textcnn = TextCNNEncoder(kernel_size, num_channel)
text_emb = textcnn(text)

from model import UtteranceEmoClf

# set random seed
random.seed(0)
torch.manual_seed(0)
# set dropout rate
drop_rate = 0.5

clf = UtteranceEmoClf(kernel_size, num_channel, drop_rate)

from train import get_loss_fn, calculate_loss

B, T, C = 16, 15, 7 # batch size, sequence length, number of classes
random.seed(0)
torch.manual_seed(0)
# create random scores and labels
scores = torch.rand(B, T, C).to(device)
label = torch.randint(-1, C, (B, T)).to(device)
# get class weights: weight = num_samples / (num_classes * num_y)
class_weights = class_stats.sum() / class_stats / class_stats.shape[0]
print(class_weights)
class_weights = class_weights.to(device)
# calculate loss
loss_fn = get_loss_fn(class_weights)
calculate_loss(loss_fn, scores, label)

from train import train_model, get_optimizer
from utils import plot_loss

clf.to(device)
optim = get_optimizer(clf, lr=1e-3, weight_decay=0)
best_model, stats = train_model(clf, val_loader, val_loader, optim, class_weights,
                                num_epoch=60, collect_cycle=20, device=device)
plot_loss(stats)


'''
import itertools
from tqdm.notebook import tqdm
from train import get_hyper_parameters

def search_param_utterance():
    """Experiemnt on different hyper parameters."""
    learning_rate, weight_decay, _ = get_hyper_parameters()
    print("learning rate from: {}\nweight_decay from: {}".format(
        learning_rate, weight_decay
    ))
    wf1 = {} # you can use this to search hyper parameters if you like
    best_model, best_stats = None, None
    best_wf1, best_lr, best_wd = 0, 0, 0
    for lr, wd in tqdm(itertools.product(learning_rate, weight_decay),
                       total=len(learning_rate) * len(weight_decay)):
        clf = UtteranceEmoClf(kernel_size, num_channel, drop_rate).to(device)
        optim = get_optimizer(clf, lr=lr, weight_decay=wd)
        model, stats = train_model(clf, trn_loader, val_loader, optim, class_weights,
                                   num_epoch=60, collect_cycle=20, device=device, verbose=False)
        # store wf1
        wf1[(lr, wd)] = stats['weighted_F1']
        # update best parameters if needed
        if stats['weighted_F1'] > best_wf1:
            best_wf1 = stats['weighted_F1']
            best_model, best_stats = model, stats
            best_lr, best_wd = lr, wd
    print("\n\nBest learning rate: {}, best weight_decay: {}".format(best_lr, best_wd))
    print("Weighted macro F-1: {:.4f}".format(best_wf1))
    print("Confusion matrix for the best model:\n", best_stats['confusion_mat'])
    plot_loss(best_stats)
    return best_model, wf1
best_utter_emo, wf1 = search_param_utterance()
'''
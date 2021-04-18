# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import random

import torch

import math

def count_lines(path):
    with open(path, 'r') as f:
        return sum([1 for _ in f])


class Example(object):

    @classmethod
    def from_tsv(cls, line, fields):
        ex = cls()
        line = line.rstrip().split('\t')
        assert len(fields) == len(line)
        for (name, field), line in zip(fields, line):
            setattr(ex, name, field.preprocess(line))
        return ex


class BaseField(object):
    def __init__(self, dtype, preprocess_fn=None, postprocess_fn=None):
        self.dtype = dtype
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def preprocess(self, x):
        if self.preprocess_fn is not None:
            return self.preprocess_fn(x)
        else:
            return x

    def process(self, batch):
        if self.postprocess_fn is not None:
            batch = self.postprocess_fn(batch)
        return torch.tensor(batch)


class TextField(BaseField):
    def __init__(self, tokenizer, dtype=torch.int64, preprocess_fn=None, postprocess_fn=None):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.dtype = dtype
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def preprocess(self, x):
        if self.preprocess_fn is not None:
            x =  self.preprocess_fn(x)
        return self.tokenizer.encode(x)

    def process(self, batch):
        if self.postprocess_fn is not None:
            batch = self.postprocess_fn(batch)
        return torch.tensor(self.pad(batch), dtype=self.dtype)

    def pad(self, batch):
        maxlen = max([len(b) for b in batch])
        padded = [b + [self.pad_id for _ in range(maxlen-len(b))] for b in batch]
        return padded


class LabelField(BaseField):
    def __init__(self, dtype, preprocess_fn=None, postprocess_fn=None):
        super(LabelField, self).__init__(dtype, preprocess_fn, postprocess_fn)

    def preprocess(self, x):
        if self.preprocess_fn is not None:
            return self.preprocess_fn(x)
        else:
            return [float(x)]

    def process(self, batch):
        if self.postprocess_fn is not None:
            batch = self.postprocess_fn(batch)
        return torch.tensor(batch, dtype=self.dtype)


class Dataset(object):
    def __init__(self, path_to_file, fields, filter_pred=None):
        self.fields = fields
        n_lines = count_lines(path_to_file)
        with open(path_to_file, 'r') as f:
            examples = [Example.from_tsv(line, fields) 
                        for line in tqdm(f, total=n_lines)]
        if filter_pred is not None:
            examples = list(filter(filter_pred, examples))
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    @classmethod
    def splits(cls, path, fields, train=None, valid=None, test=None, **kwargs):

        train_data = None if train is None \
            else cls(os.path.join(path, train), fields, **kwargs)

        val_data = None if valid is None \
            else cls(os.path.join(path, valid), fields, **kwargs)

        test_data = None if test is None \
            else cls(os.path.join(path, test), fields, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class Iterator(object):
    def __init__(self, dataset, batch_size, shuffle=True, repeat=False):
        self.dataset = dataset
        self.bsz = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.n_updates = 0
    
    def __len__(self):
        return math.ceil(len(self.dataset.examples)/self.bsz)

    def __iter__(self):
        while True:
            self.init_epoch()
            for i in range(0, len(self.dataset), self.bsz):
                yield Batch(self.dataset.examples[i:i+self.bsz], self.dataset)
                self.n_updates += 1
            if not self.repeat:
                return

    def init_epoch(self):
        if self.shuffle:
            self.dataset.examples = random.sample(self.dataset.examples, len(self.dataset))

    @classmethod
    def splits(cls, datasets, batch_size, **kwargs):
        return tuple(cls(datasets[i], batch_size, **kwargs) for i in range(len(datasets)))
            


class Batch(object):
    def __init__(self, data=None, dataset=None):
        self.batch_size = len(data)
        self.dataset = dataset

        for name, field in dataset.fields:
            batch = [getattr(x, name) for x in data]
            setattr(self, name, field.process(batch))

    def __len__(self):
        return self.batch_size
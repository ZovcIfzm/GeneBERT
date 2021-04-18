'''
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    get_linear_schedule_with_warmup,
)
'''
'''
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = RobertaTokenizerFast.from_pretrained('DeepInflam') # do_lower_case=True
tokens = tokenizer.tokenize(sentenceA)
tokenizer.convert_tokens_to_ids(tokens)

max_len = 0

# For every sentence...
for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
'''
'''
device = torch.device('cpu')

# Load pretrained model and tokenizer
config = RobertaConfig.from_pretrained(
    args.model_name_or_path,
    num_labels=args.num_labels,
)

tokenizer = RobertaTokenizerFast.from_pretrained(
    args.model_name_or_path
)

model = RobertaForMaskedLM.from_pretrained(
    args.model_name_or_path,
    config=config,
)


model.to(device)

tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputA))
tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputB))
label = inputC

'''

import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

MODEL_DIR = "DeepInflam_1500"
model_class, tokenizer_class, pretrained_weights = (ppb.RobertaForMaskedLM, ppb.RobertaTokenizerFast, MODEL_DIR)

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

from transformers import pipeline
fill_mask = pipeline(
    "fill-mask",
    model=MODEL_DIR,
    tokenizer=MODEL_DIR
)

print(fill_mask("hll hrr <mask>."))

df = pd.DataFrame({"A": ["hhhh llhl <mask>", "hhlmmm mmmmm <mask>"]})

#df2 = pd.read_csv('./word_tokenized/C1Test.csv', delimiter='\t', header=None)

#print(df2.head())

tokenized = df["A"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#padd
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print(padded.shape)
attention_mask = np.where(padded != 0, 1, 0)

#input
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids)
    features = last_hidden_states[0][:,0,:].numpy()

print(features)

'''
df = pd.read_csv('./sample_data/Train.tsv', delimiter='\t', header=None)

tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#padd
max_len = 1500
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print(padded.shape)
attention_mask = np.where(padded != 0, 1, 0)

#input
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids)
    features = last_hidden_states[0][:,0,:].numpy()

print(features)
'''
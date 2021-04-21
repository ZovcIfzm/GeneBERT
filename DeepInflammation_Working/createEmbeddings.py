

import argparse
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

'''
python3 createEmbeddings.py --cell_file ./word_tokenized/C1Train.csv --output_file C1TrainEmbeddings.csv
python3 createEmbeddings.py --cell_file ./word_tokenized/C2Train.csv --output_file C2TrainEmbeddings.csv
python3 createEmbeddings.py --cell_file ./word_tokenized/C1Valid.csv --output_file C1ValidEmbeddings.csv
python3 createEmbeddings.py --cell_file ./word_tokenized/C2Valid.csv --output_file C2ValidEmbeddings.csv

python3 createEmbeddings.py --cell_file ./word_tokenized/DiffTrain.csv --output_file ./embeddings/DiffTrainEmbeddings.csv
python3 createEmbeddings.py --cell_file ./word_tokenized/DiffValid.csv --output_file ./embeddings/DiffValidEmbeddings.csv
python3 createEmbeddings.py --cell_file ./word_tokenized/DiffTest.csv --output_file ./embeddings/DiffTestEmbeddings.csv
'''



def createEmbeddings(cell_file, output_file):
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

    df2 = pd.read_csv(cell_file, delimiter='\t', header=None)

    print(df2.head())

    df2['vals'] = df2[0].map(lambda x: x[694:])

    print(df2["vals"].head())

    tokenized = df2["vals"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
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

    #print(features)

    np.savetxt(output_file, 
            features,
            delimiter =", ", 
            fmt ='% s')

parser = argparse.ArgumentParser(description='CreateLabels')
parser.add_argument('--cell_file', type=str, default='', help='Path to file for cell 1')
parser.add_argument('--output_file', type=str, default="output_cell.csv", help="path for output file")

args = parser.parse_args()

createEmbeddings(args.cell_file, args.output_file)

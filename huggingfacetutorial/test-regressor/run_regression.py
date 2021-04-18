# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict
from tqdm import tqdm

import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

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

from dataset import (
    TextField,
    LabelField,
    Dataset,
    Iterator,
)


def main(args):

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

    text_field = TextField(tokenizer)
    label_field = LabelField(torch.long if args.num_labels > 1  else torch.float)

    if args.do_test:
        fields = [('src', text_field), ('ref', text_field)]
    else:
        fields = [('src', text_field), ('ref', text_field), ('score', label_field)]

    # Training
    if args.do_train:
        # setup dataset
        print('Loading training data ...')
        train_data = Dataset(
            path_to_file=args.data,
            fields=fields
        )

        train_iter = Iterator(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            repeat=False,
        )

        train(args, train_iter, model, device)
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)


    # Evaluaiton
    if args.do_eval:
        model.eval()

        # setup dataset
        print('Loading development data ...')
        valid_data = Dataset(
            path_to_file=args.data,
            fields=fields
        )

        valid_iter = Iterator(
            dataset=valid_data,
            batch_size=args.batch_size,
            shuffle=True,
            repeat=False,
        )

        preds_list = [] 
        refs_list = []

        print("DEBUG::tokeniszerlength", len(tokenizer))
        #model.resize_token_embeddings(len(tokenizer))
        print("DEBUG::", len(valid_iter))
        for batch in tqdm(valid_iter, total=len(valid_iter)):
            input_ids = torch.cat([batch.src, batch.ref[:, 1:]], dim=1).to(device)
            labels = batch.score.squeeze(1).to(device)

            token_type_ids = [
                torch.zeros_like(batch.src),
                torch.ones_like(batch.ref[:, 1:])
            ]

            token_type_ids = torch.cat(token_type_ids, dim=1).to(device)
            print("DEBUG::", input_ids.shape, token_type_ids.shape, labels.shape)
            outputs = model(input_ids, token_type_ids=token_type_ids, labels=labels)

            print("DEBUG:: made it past")
            preds = torch.ge(outputs, args.threshold).int().squeeze(1)

            preds_list.append(preds.to('cpu'))
            refs_list.append(labels.int().to('cpu'))


        preds_list = torch.cat(preds_list)
        refs_list = torch.cat(refs_list)

        avg = 'macro' if args.num_labels > 1 else 'micro'
        precision = precision_score(refs_list, preds_list, average=avg)
        recall = recall_score(refs_list, preds_list, average=avg)
        f1 = f1_score(refs_list, preds_list, average=avg)

        print(f"Presion: {precision * 100}", end='\t')
        print(f"Recall: {recall * 100}", end='\t')
        print(f"F1 score: {f1 * 100}")


    if args.do_test:
        model.eval()

        # setup dataset
        print('Loading test data ...')
        test_data = Dataset(
            path_to_file=args.data,
            fields=fields
        )

        test_iter = Iterator(
            dataset=test_data,
            batch_size=args.batch_size,
            shuffle=True,
            repeat=False,
        )
 
        for batch in tqdm(test_iter, total=len(test_iter)):
            input_ids = torch.cat([batch.src, batch.ref[:, 1:]], dim=1).to(device)

            token_type_ids = [
                torch.zeros_like(batch.src),
                torch.ones_like(batch.ref[:, 1:])
            ]
            token_type_ids = torch.cat(token_type_ids, dim=1).to(device)
            outputs = model(input_ids, token_type_ids=token_type_ids)[0]

            for src, ref, out in zip(batch.src, batch.ref, outputs):
                src = src[1:src.tolist().index(tokenizer.sep_token_id)]
                ref = ref[1:ref.tolist().index(tokenizer.sep_token_id)]
                src = tokenizer.decode(src)
                ref = tokenizer.decode(ref)
                if args.num_labels > 1:
                    out = torch.argmax(out)
                print(src + '\t' + ref + '\t' + str(out.item()))


def train(args, train_iter, model, device):
    # setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    max_iteration = len(train_iter) * args.max_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_iteration,
    )

    for epoch in range(args.max_epoch):
        model.train()
        with tqdm(train_iter, dynamic_ncols=True) as pbar:
            for batch in pbar:
                input_ids = torch.cat([batch.src, batch.ref[:, 1:]], dim=1).to(device)
                token_type_ids = [
                    torch.zeros_like(batch.src),
                    torch.ones_like(batch.ref[:, 1:])
                ]
                token_type_ids = torch.cat(token_type_ids, dim=1).to(device)
                labels = batch.score.to(device)
                
                print("DEBUG::Train", input_ids.shape, token_type_ids.shape, labels.shape)
                #print("DEBUG::MAX", max(input_ids), max(token_type_ids), max(labels))
                print("DEBUG::actual", input_ids)
                outputs = model(input_ids, token_type_ids=token_type_ids, labels=labels)
                loss, logits = outputs[:2]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # setting of progressbar
                progress_state = OrderedDict(
                    loss=loss.item(),
                    bsz=len(batch),
                    num_updates=train_iter.n_updates)
                pbar.set_postfix(progress_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    '''
    Implementation of BERT regressor by HuggingFace's transformers
    '''
    )

    # Required
    parser.add_argument("--model_type", default=None, type=str, required=True,
        help="")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
        help="")
    parser.add_argument("--num_labels", default=None, type=int, required=True,
        help="Number of output classes")
    parser.add_argument("--save_dir", default=None, type=str, 
        help="The output directory where the model predictions and checkpoints "
             "will be written.")

    # Dataset
    parser.add_argument("--data",  type=str,
        default="./sample_data/Train.tsv",
        help="file name of data")
    parser.add_argument("--batch_size", default=32, type=int,
        help="batch size")
    parser.add_argument("--src_min", default=0, type=int,
        help="minimum sentence length of source side")
    parser.add_argument("--ref_min", default=0, type=int,
        help="minimum sentence length of reference side")
    parser.add_argument("--src_max", default=128, type=int,
        help="maximum sentence length of source side")
    parser.add_argument("--ref_max", default=128, type=int,
        help="maximum sentence length of reference side")

    # Training
    parser.add_argument("--max_epoch", default=3, type=int,
        help="maximum step of iteration in training")
    parser.add_argument("--warmup_steps", default=0, type=int,
        help="warmup step")
    parser.add_argument("--lr", default=2e-5, type=float,
        help="learning rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
        help="Max gradient norm.")

    # Evaluation
    parser.add_argument("--threshold", type=float, default=0.5,
        help="Threshold of regresssion")

    # Others
    parser.add_argument("--do_train", action="store_true", 
        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
        help="Whether to run eval on the eval set.")
    parser.add_argument("--do_test", action="store_true",
        help="Whether to run test on the test set.")
 
    args = parser.parse_args()
    main(args)
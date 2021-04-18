import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import numpy as np

class TextCNNEncoder(nn.Module):
    """Text encoder using CNN"""
    def __init__(self, kernel_size, num_channel):
        """
        Input:
            - kernel_size: a list for size of the kernels. e.g. [3, 4, 5] means we
                will have three kernels with size 3, 4, and 5 respectively.
            - num_channel: number of output channels for each kernel.

        A few key steps of the network:
            conv -> relu -> global max pooling -> concatenate
        
        Hint:
            - Use k // 2 as padding in cnn.
            - nn.ModuleList() is useful.
            - You don't need to do specific initialization, we can use the
                default initialization.
        """
        
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(len(kernel_size)*num_channel, num_channel, kernel_size[i]) for i in range(len(kernel_size))])
        self.fc = nn.Linear(len(kernel_size)*num_channel, len(kernel_size)*num_channel)
        self.num_channel = num_channel
    
    def forward(self, text_emb):
        """
        Input:
            - text_emb: input utterances with shape (N, L, 300), where N is the
                number of utterances in a batch, L is the longest utterance.
                Note we concatenate utterances from all dialogues.
        Return:
            - output: encoded utterances with shape (N, len(kernel_size) * num_channel)
        """
        
        # TextCNN forward
        text_emb = text_emb.permute(0,2,1)
        x = [conv(text_emb) for conv in self.convs]
        pool_x = [F.max_pool1d(F.relu(conved), conved.size(2)) for conved in x]
        fc_x = torch.cat([torch.flatten(pool, start_dim=1) for pool in pool_x], dim=1)
        output = self.fc(fc_x)

        return output


class UtteranceEmoClf(nn.Module):
    """Single utterance emotion classifier."""
    def __init__(self, kernel_size, num_channel, drop_rate):
        """
        Input:
            - kernel_size: a list for size of the kernels for CNN. e.g. [3, 4, 5]
                means we will have three kernels with size 3, 4, and 5 respectively.
            - num_channel: number of output channels for CNN.
            - drop_rate: dropout rate.
        
        A few key steps of the network:
            textcnn -> concat text_emb and audio_emb -> dropout -> fully connected layer
        """
        super().__init__()
        self.textcnn = TextCNNEncoder(kernel_size, num_channel)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(1611+len(kernel_size)*num_channel, 7)

    def forward(self, text_emb, audio_emb, num_utt):
        """
        Input:
            - text_emb: input utterances with shape (N, L, 300), where N is the
                number of utterances in a batch, L is the longest utterance.
                Note we concatenate utterances from all dialogues.
            - audio_emb: audio embedding with shape (B, T, 1611), where B is
                the batch size (number of dialogues), T is sequence length
                (max number of utterances), 1611 is the number of features
                for audio embedding.
            - num_utt: list, stores the number of utterances in each dialogue.
        Return:
            - output: (B, T, 7), where 7 is the number of emotions we want to classify.
                This stores the scores for each emotion before softmax layer.
        """

        new_text_emb = self.textcnn(text_emb)
        split = torch.split(new_text_emb, num_utt)
        
        new_seq = pad_sequence(split)
        new_seq = torch.transpose(new_seq, 0, 1)
   
        new_emb = torch.cat((new_seq, audio_emb), dim=2)

        output = self.fc(self.dropout(new_emb))

        # calculate utterance embedding using TextCNN

        # reshape text_emb to appropriate shape so that we can concatenate it with audio_emb
        # Hint: use torch.split() to split text_emb, then use pad_sequence()
        # to pad to the same length

        # concatenate text_emb and audio_emb

        # apply dropout

        # linear layer

        return output


class ContextGRU(nn.Module):
    """One layer GRU that utilizes dialogue context to classify utterance emotion."""
    def __init__(self, input_dim, hidden_dim, drop_rate, bidirectional):
        """
        Input:
            - input_dim: dimension of input features.
            - hidden_dim: dimension of hidden state.
            - drop_rate: dropout rate.
            - bidirectional: whether to use bidirectional GRU or not.
        
        A few key steps of the network:
            gru -> dropout -> fully connected layer
        """
        super().__init__()

    def forward(self, x, num_utt):
        """
        Input:
            - x: input features with shape (B, T, input_dim), where B is batch
                size (number of dialogues), T is sequence length (max number of utterances).
            - num_utt: list, stores the number of utterances in each dialogue.
        Return:
            - output: (B, T, 7), where 7 is the number of emotions we want to classify.
                This stores the scores for each emotion before softmax layer.
        """
        output = None

        # pack the padded sequence
        # Hint: pay attention to arguments batch_first and enforce_sorted
        
        # run GRU
        
        # pad the packed sequence
        
        # apply dropout
        
        # linear layer
        
        return output


class ContextEmoClf(nn.Module):
    """Context aware emotion classifier."""
    def __init__(self, kernel_size, num_channel, hidden_dim, drop_rate, bidirectional):
        """
        Input:
            - kernel_size: a list for size of the kernels for CNN. e.g. [3, 4, 5]
                means we will have three kernels with size 3, 4, and 5 respectively.
            - num_channel: number of output channels for CNN.
            - hidden_dim: dimension of hidden state for GRU.
            - drop_rate: dropout rate.
            - bidirectional: whether to use bidirectional GRU or not.
        
        A few key steps of the network:
            textcnn -> concat text_emb and audio_emb -> dropout -> context_gru
        """
        super().__init__()

    def forward(self, text_emb, audio_emb, num_utt):
        """
        Input:
            - text_emb: input utterances with shape (N, L, 300), where N is the
                number of utterances in a batch, L is the longest utterance.
                Note we concatenate utterances from all dialogues.
            - audio_emb: audio embedding with shape (B, T, 1611), where B is
                the batch size (number of dialogues), T is sequence length
                (max number of utterances), 1611 is the number of features
                for audio embedding.
            - num_utt: list, stores the number of utterances in each dialogue.
        Return:
            - output: (B, T, 7), where 7 is the number of emotions we want to classify.
                This stores the scores for each emotion before softmax layer.
        """
        output = None

        # calculate utterance embedding using TextCNN

        # reshape text_emb to appropriate shape so that we can concatenate it with audio_emb
        # Hint: use torch.split() to split text_emb, then use pad_sequence()
        # to pad to the same length

        # concatenate text_emb and audio_emb

        # apply dropout

        # calculate scores using ContextGRU

        return output

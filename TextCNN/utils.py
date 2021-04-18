import bisect

import torch
import matplotlib.pyplot as plt


def plot_loss(stats):
    """Plot training loss and validation loss."""
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()


def meld_collate_fn(batch):
    """Collate function for dataloader."""
    # audio embedding
    audio_emb = [item[1] for item in batch]
    audio_emb = torch.nn.utils.rnn.pad_sequence(audio_emb, batch_first=True)
    label = [item[2] for item in batch]
    num_utterances = [i.shape[0] for i in label]
    # labels
    label = torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=-1)
    # text embedding
    text_emb = []
    for dialogue in batch:
        text_emb += dialogue[0]
    text_emb = torch.nn.utils.rnn.pad_sequence(text_emb, batch_first=True)

    return text_emb, audio_emb, label, num_utterances


def reset_dialogue_id(df, missing_ids):
    """
    Reset Dialogue_ID in df so that Dialogue_ID ranges from 0 to max_id - 1
    where max_id is the number of unique Dialogue_ID in df.
    Input:
        - df: pandas dataframe to work on
        - missing_ids: list of missing Dialogue_ID
    Return:
        - df after resetting Dialogue_ID
    """
    if len(missing_ids) > 0:
        for i in range(df.shape[0]):
            idx = bisect.bisect_left(missing_ids, df.loc[i, 'Dialogue_ID'])
            df.loc[i, 'Dialogue_ID'] -= idx
    return df


def reset_audio_emb(audio_emb, missing_ids):
    """
    Reset keys in audio_emb.
    Input:
        - audio_emb: dict to work on
        - missing_ids: list of missing Dialogue_ID
    Return:
        - audio_emb after resetting
    """
    if len(missing_ids) > 0:
        # sort by Dialogue_ID
        keys = sorted(list(audio_emb.keys()), key=lambda x: int(x.split('_')[0]))
        for key in keys:
            ids = key.split('_')
            idx = bisect.bisect_left(missing_ids, int(ids[0]))
            if idx > 0:
                new_key = str(int(ids[0]) - idx) + '_' + ids[1]
                audio_emb[new_key] = audio_emb[key]
                del audio_emb[key]
    dialogue_ids = set([int(i.split('_')[0]) for i in audio_emb.keys()])
    assert len(dialogue_ids) == max(dialogue_ids) + 1
    return audio_emb

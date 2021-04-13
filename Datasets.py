import torch
import torchtext
import os

def get_dataset(dataset_name, tokenizer, dataset_path=None):
    train, val, test = None, None, None

    # prepare data and get iterator
    TEXT = torchtext.data.Field(use_vocab=False, tokenize=tokenizer.encode, pad_token=tokenizer.pad_token_id)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    # need to set newline_eos to be false, otherwise add eos in the end
    if dataset_name == 'wiki2':
        train, val, test = torchtext.datasets.WikiText2.splits(TEXT, newline_eos=False)
    elif dataset_name == 'ptb':
        train, val, test = torchtext.datasets.PennTreebank.splits(TEXT, newline_eos=False)
    elif dataset_name == 'wiki103':
        train, val, test = torchtext.datasets.WikiText103.splits(TEXT, newline_eos=False) 
    else:
        train = torchtext.datasets.LanguageModelingDataset(dataset_path, TEXT, newline_eos=False)

    return train, val, test

def get_iter(train, val=None, test=None, batch_size=1, bptt_len=512):
    val_iter, test_iter = None, None
    if val and test:
        train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test), batch_size=batch_size, bptt_len=bptt_len)
    else:
        train_iter = torchtext.data.BPTTIterator(train, batch_size=batch_size, bptt_len=bptt_len)

    return train_iter, val_iter, test_iter
    

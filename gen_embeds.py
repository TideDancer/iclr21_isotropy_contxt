import torch
import numpy as np
import pickle
import os
import logging

import Args
import Datasets
import Models

args, device = Args.init()
model, tokenizer = Models.get_model_tokenizer(args.model)
model.to(device)

args.save_file = args.dataset + '/' + args.save_file
model.eval()
args.save_file = 'embeds/' + args.save_file

# get dataset and iterator
train, val, test = Datasets.get_dataset(args.dataset, tokenizer, dataset_path=args.datapath)
train_iter, val_iter, test_iter = Datasets.get_iter(train, val=val, test=test, batch_size=args.batch_size, bptt_len=args.bptt_len)

# init dict 
token_dict = {} 

# big loop
for batch_idx, data in enumerate(train_iter):
    # if uniform sampling
    if args.sample < 1:
        if np.random.rand() < float(args.sample):
            continue

    logging.info("batch idx: "+str(batch_idx))

    if data.text.t().shape[1] < args.bptt_len:
        logging.info('skip')
        continue

    # forward computing
    input_ids = data.text.t().to(device)
    outputs = model(input_ids)
    if args.model == 'dist' or args.model == 'gpt' or args.model == 'xlm':
        hidden_states = outputs[1][1:]
    else:
        hidden_states = outputs[2][1:]

    # obtain layer's hidden states:
    hidden_y = hidden_states[args.layer] # batch x len x dim

    for x_id in range(args.bptt_len):
        y = hidden_y[:, x_id]
        cnt = 0
        for b in range(args.batch_size):
            key = input_ids[b, x_id].detach().cpu().item()
            token = tokenizer.convert_ids_to_tokens(key)
            token_dict[token] = token_dict.get(token, []) + [(y[cnt].detach().cpu().numpy(), batch_idx, x_id, key)] # d[token] = [(embedding, batch_idx, x_id, key)]
            cnt += 1

pickle.dump((batch_idx, token_dict), open(args.save_file, 'wb'))

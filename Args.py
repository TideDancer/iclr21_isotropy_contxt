import argparse
import logging
import torch
import sys

def init(prj_name='Experiments'):
    parser = argparse.ArgumentParser(description=prj_name)

    # required
    parser.add_argument('model', type=str, help='model: gpt, gpt2, bert, dist (distilbert), xlm')
    parser.add_argument('dataset', type=str, help='dataset: wiki2, ptb, wiki103, or other customized datapath')
    parser.add_argument('layer', type=int, help='layer id')

    # optional
    parser.add_argument('--save_file', type=str, default='tmp', help='save pickle file name')
    parser.add_argument('--log_file', type=str, default=None, help='log file name')
    parser.add_argument('--datapath', type=str, default=None, help='customized datapath')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, default=1')
    parser.add_argument('--bptt_len', type=int, default=512, help='tokens length, default=512')
    parser.add_argument('--sample', type=float, default=1, help='[beta], uniform with probability=beta')
    parser.add_argument('--no_cuda', action="store_true", help='disable gpu')

    args = parser.parse_args()

    if args.log_file is not None:
        logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.info(args)

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    
    return args, device


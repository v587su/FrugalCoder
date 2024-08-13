import argparse
import torch
import numpy
import re
import datasets
import pickle
from collections import Counter
from nltk.util import ngrams
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, NamedTuple
from bleu_ignoring import sentence_bleu, SmoothingFunction

def wrap_code(code,code_id, with_bracket=True):
    code = re.sub(r'(?<=[\S])([ ]{4,})(?=[\S])', r'\n\1', code)
    code = re.sub(r'(?<=[\S])([\t]{1,})(?=[\S])', r'\n\1', code)
    pre_fix = f'public class id_{str(code_id)} {{\n'
    post_fix = '\n}\n' if with_bracket else ''
    return pre_fix+code+post_fix



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_dev', action='store_true', help='development mode') 
    parser.add_argument('--load_prediction', action='store_true', help='development mode') 
    parser.add_argument('--mode', type=str, default='train', help='development mode') 
    parser.add_argument('--exp_name', type=str, default='tosem', help='development mode') 
    parser.add_argument('--run_name', type=str, help='development mode') 
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Path to the dataset') 
    parser.add_argument('--chunk', type=int, default=-1, help='Path to the dataset') 
    parser.add_argument('--dataset_name', type=str, default='gpt2_score_seq_prob', help='Path to the dataset') 
    parser.add_argument('--type', type=str, default='seq', help='Path to the dataset') 
    parser.add_argument('--model', type=str, default='gpt2', help='Path to the dataset') 
    parser.add_argument('--language', type=str, default='java', help='Path to the dataset') 
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch Number')
    parser.add_argument('--text_length', type=int, default=128, help='Length of the text')
    parser.add_argument('--min_query_len', type=int, default=10, help='Length of the text')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Length of the text')
    parser.add_argument('--output_dir', type=str, default=None, help='Length of the text')
    parser.add_argument('--estimator', type=str, default=None, help='Length of the text')
    parser.add_argument('--cache_path', type=str, default="./cached/gpt2", help='Length of the text')
    parser.add_argument('--device', type=str, default="cuda:0", help='Length of the text')
    parser.add_argument('--metric', type=str, default="nll_loss", help='Length of the text')
    parser.add_argument('--deepspeed', type=str, default="ds_config.json", help='Length of the text')
    parser.add_argument('--local_rank', type=int, default=-1, help='Length of the text')
    return parser.parse_args()



class CBleu:
    def __init__(self, corpus):
        MAXN = 4
        all_ngrams = []
        for item in corpus:
            for i in range(1, MAXN+1):
                n_grams = list(ngrams(item, i))
                all_ngrams.extend(n_grams)
        freq = Counter(all_ngrams)
        most_common_dict = dict(freq.most_common(150))
        self.most_common_dict = most_common_dict
        self.smfunc = SmoothingFunction(epsilon=0.0001).method2
    
    def sentence_bleu(self,references, hypothesis):
        return sentence_bleu([references],hypothesis,smoothing_function=self.smfunc, ignoring=self.most_common_dict)


import os
def get_latest_checkpoint(dir_path):
    steps = []
    for file in os.listdir(dir_path):
        if file.startswith('checkpoint'):
            steps.append(int(file.split('-')[-1]))
    if len(steps) > 0:
        return os.path.join(dir_path, f'checkpoint-{max(steps)}')
    else:
        return None

def get_latest_lighting_checkpoint(dir_path):
    steps = []
    files = []
    versions = []
    for file in os.listdir(dir_path):
        if file.startswith('version'):
            versions.append(int(file.split('_')[-1]))
    for file in os.listdir(os.path.join(dir_path, f'version_{max(versions)}', 'checkpoints')):
        if file.startswith('epoch'):
            steps.append(int(file.split('=')[-1].split('.')[0]))
            files.append(file)
    if len(steps) > 0:
        return os.path.join(dir_path, f'version_{max(versions)}', 'checkpoints', files[steps.index(max(steps))])
    else:
        return None

def load_dataset(args, name, need_score=False):
    dataset = datasets.load_from_disk(os.path.join(args.data_dir, args.exp_name, name, args.language, args.model))
    scores = pickle.load(open(os.path.join(args.data_dir, args.exp_name, name, f'{args.language}_{args.model}_scores.pkl'), 'rb'))
    dataset = dataset.add_column('label', scores[args.metric])
    if need_score:
        return dataset, scores[args.metric]
    return dataset
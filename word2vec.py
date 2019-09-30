import collections
import math
import random
import zipfile

with zipfile.ZipFile('./data/ptb.zip', 'r') as zin:
    zin.extractall('./data/')

def get_raw_dataset(filename = './data/ptb/ptb.train.txt'):
    with open(filename, 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]
    return raw_dataset

def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)

def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i-window_size), min(len(st), center_i+1+window_size)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


train_file = './data/ptb/ptb.train.txt'
raw_dataset = get_raw_dataset(train_file)

counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x:x[1]>=5, counter.items()))

idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
           for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

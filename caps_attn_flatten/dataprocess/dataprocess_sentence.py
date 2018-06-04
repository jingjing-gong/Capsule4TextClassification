import numpy as np
import _pickle as pkl
import os, operator
from collections import defaultdict
from tensorflow.python.util import nest
from vocab import Vocab
import argparse

parser = argparse.ArgumentParser(description="datasets")

parser.add_argument('--train-set', action='store', dest='train_set', default=None)
parser.add_argument('--dev-set', action='store', dest='dev_set', default=None)
parser.add_argument('--test-set', action='store', dest='test_set', default=None)
parser.add_argument('--ref-embedding', action='store', dest='ref_emb', default='/home/jjgong/data/glove300d/glove.840B.300d.txt')
parser.add_argument('--dest-dir', action='store', dest='dest_dir', default='./')
parser.add_argument('--label2id', action='store', dest='label2id', default=None)
parser.add_argument('--base-wd-freq', type=int, action='store', dest='base_wd_freq', default=3)
parser.add_argument('--labelshift', type=int, action='store', dest='shift', default=1)

args = parser.parse_args()

def extract(fn):
    label_collect = []
    snt_tok_collect = []
    with open(fn, 'r') as fd:
        for line in fd:
            item = line.strip().split('\t\t')
            try:
                label = item[0]
                snt = item[1]
            except:
                print(line)
                print(item)
                raise ValueError

            snt2wd = snt.strip().split(' ')
            label_collect.append(label)
            snt_tok_collect.append(snt2wd)
    return label_collect, snt_tok_collect

def constructLabel_dict(labels, savepath):
    label_freq = defaultdict(int)
    for i in labels:
        label_freq[i] += 1
    class_num = len(label_freq.values())
    id2revfreq = {}
    dinominator = float(sum(label_freq.values()))
    if args.label2id is None:
        label2id = dict(list(zip(label_freq.keys(), [int(o)-args.shift for o in label_freq.keys()])))
    else:
        with open(args.label2id, 'rb') as fd:
            label2id = pkl.load(fd)
    id2label = {idx: label for label, idx in label2id.items()}
    for item in id2label:
        label = id2label[item]
        freq = label_freq[label]
        id2revfreq[item] = float(dinominator)/float(freq)
    dino = float(sum(id2revfreq.values()))
    id2weight = {idx: class_num * revfreq/dino for idx, revfreq in id2revfreq.items()}

    with open(savepath, 'wb') as fd:
        pkl.dump(label2id, fd)
        pkl.dump(id2label, fd)
        pkl.dump(id2revfreq, fd)
        pkl.dump(id2weight, fd)

def loadLabel_dict(savepath):
    with open(savepath, 'rb') as fd:
        label2id = pkl.load(fd)
        id2label = pkl.load(fd)
        id2revfreq = pkl.load(fd)
        id2weight = pkl.load(fd)
    return label2id, id2label, id2revfreq, id2weight


def readEmbedding(fileName):
    """
    Read Embedding Function

    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'r') as f:
        for line in f:
            line_uni = line.strip()
            values = line_uni.split(' ')
            if len(values) != 301:
                continue
            word = values[0]
            w2v_line = ' '.join(values)
            embeddings_index[word] = w2v_line
    return embeddings_index

def buildEmbedding(src_embed_file, tgt_embed_file, word_dict):
    emb_dict = readEmbedding(src_embed_file)
    with open(tgt_embed_file, 'w') as fd:
        for word in word_dict:
            if word in emb_dict:
                fd.writelines(emb_dict[word]+'\n')
    return None

if __name__ == '__main__':
    vocab = Vocab()
    tok_collect = []
    labels_collect = []
    if args.train_set:
        train_label, train_toks = extract(args.train_set)
        tok_collect.append(train_toks)
        labels_collect.append(train_label)
    if args.dev_set:
        dev_label, dev_toks = extract(args.dev_set)
        tok_collect.append(dev_toks)
        labels_collect.append(dev_label)
    if args.test_set:
        test_label, test_toks = extract(args.test_set)

    vocab.construct(nest.flatten(tok_collect))
    vocab.limit_vocab_length(base_freq=args.base_wd_freq)
    vocab.save_vocab(os.path.join(args.dest_dir, 'vocab.pkl'))

    constructLabel_dict(nest.flatten(labels_collect), os.path.join(args.dest_dir, 'label2id.pkl'))

    vocab = Vocab()
    vocab.load_vocab_from_file(os.path.join(args.dest_dir, 'vocab.pkl'))

    buildEmbedding(args.ref_emb, os.path.join(args.dest_dir, 'embedding.txt'), vocab.word_to_index)

    label2id, id2label, id2revfreq, id2weight = loadLabel_dict(os.path.join(args.dest_dir, 'label2id.pkl'))

    if args.train_set:
        train_label = [label2id[o] for o in train_label]
        train_toks = nest.map_structure(lambda x: vocab.encode(x), train_toks)
        train_set = [o for o in zip(train_label, train_toks)]
        with open(os.path.join(args.dest_dir, 'trainset.pkl'), 'wb') as fd:
            pkl.dump(train_set, fd)
    if args.dev_set:
        dev_label = [label2id[o] for o in dev_label]
        dev_toks = nest.map_structure(lambda x: vocab.encode(x), dev_toks)
        dev_set = [o for o in zip(dev_label, dev_toks)]
        with open(os.path.join(args.dest_dir, 'devset.pkl'), 'wb') as fd:
            pkl.dump(dev_set, fd)
    if args.test_set:
        test_label = [label2id[o] for o in test_label]
        test_toks = nest.map_structure(lambda x: vocab.encode(x), test_toks)
        test_set = [o for o in zip(test_label, test_toks)]
        with open(os.path.join(args.dest_dir, 'testset.pkl'), 'wb') as fd:
            pkl.dump(test_set, fd)




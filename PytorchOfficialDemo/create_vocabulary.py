import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from vocabulary import Voc

from config import SENTENCE_MAX_LENGTH, DATA_DIR, WORD_MIN_FREQUENCY


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # 变成小写、去掉前后空格，然后unicode变成ascii
    s = unicode_to_ascii(s.lower().strip())
    # 在标点前增加空格，这样把标点当成一个词
    s = re.sub(r"([.!?])", r" \1", s)
    # 字母和标点之外的字符都变成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # 因为把不用的字符都变成空格，所以可能存在多个连续空格
    # 下面的正则替换把多个空格变成一个空格，最后去掉前后空格
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def read_vocs(data_file, corpus_name):
    lines = open(data_file, encoding='utf-8').read().strip().split('\n')
    # 每行用tab切分成问答两个句子
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)

    return voc, pairs


def filter_pair(p):
    return len(p[0].split(' ')) < SENTENCE_MAX_LENGTH and len(p[1].split(' ')) < SENTENCE_MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(data_file, corpus_name):
    print("Start preparing training data ...")
    voc, pairs = read_vocs(data_file, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)

    return voc, pairs


def trim_rare_words(voc, pairs):
    voc.trim(WORD_MIN_FREQUENCY)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break

        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


def create_voc():
    corpus_name = "cornell_movie_dialogs_corpus"
    data_file = os.path.join(DATA_DIR, 'formatted_movie_lines.txt')
    voc, pairs = load_prepare_data(data_file, corpus_name)
    pairs = trim_rare_words(voc, pairs)

    return voc, pairs


if __name__ == "__main__":
    create_voc()

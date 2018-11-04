# coding:utf-8

import os
import torch
import spacy
spacy_en = spacy.load('en')


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seq_mask(seq_len, device, max_len=None):
    batch_size = seq_len.size(0)
    if not max_len:
        max_len = torch.max(seq_len)
    mask = torch.zeros((batch_size, max_len), device=device)
    for i in range(batch_size):
        for j in range(seq_len[i]):
            mask[i][j] = 1
    return mask


def get_wordoverlaps(s1, s2):
    '''
    计算句子s1,s2的词重叠向量
    '''
    s1_overlap = []
    s2_overlap = []
    for word in s1:
        if word in s2:
            s1_overlap.append(1)
        else:
            s1_overlap.append(0)

    for word in s2:
        if word in s1:
            s2_overlap.append(1)
        else:
            s2_overlap.append(0)

    return s1_overlap, s2_overlap

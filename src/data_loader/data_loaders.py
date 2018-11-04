# coding:utf-8

import os
import pickle

import torchtext.data as data
import torchtext.vocab as vocab
import numpy as np
from base import BaseNlpDataLoader


class WikiqaDataLoader(BaseNlpDataLoader):
    """
    WikiQA data loading process.
    """
    def __init__(self, tokenizer, data_path, train_batch_size, eval_batch_size, train_file=None, eval_file=None, test_file=None,
                 pretrain_ebd_file=None, shuffle=True, lower=True, device=-1):
        '''
        :param data_path: Data path.
        :param train_batch_size: Train batch size.
        :param eval_batch_size: Evaluate and test batch size
        :param train_file: train file name.
        :param eval_file: eval file name.
        :param test_file: test file name.
        :param pretrain_ebd_file: pretrain embedding file
        :param ebd_cache: embedding cache file.
        :param shuffle:  Whether to shuffle examples between epochs.
        :param lower: Whether to lowercase the text
        :param device:  A string or instance of torch.device specifying which device the Variables are going to be created on.
               If left as default, the tensors will be created on cpu. Default: None.
        '''
        super(WikiqaDataLoader, self).__init__()

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # define fields
        text = data.Field(sequential=True, tokenize=tokenizer, lower=lower, include_lengths=True)
        label = data.Field(sequential=False, use_vocab=False)

        # load data
        train_fields = {
            'question': ('question', text),
            'right_answer': ('right_answer', text),
            'wrong_answer': ('wrong_answer', text),
        }

        eval_fields = {
            'Question': ('question', text),
            'Sentence': ('answer', text),
            'Label': ('label', label)
        }

        self.train = data.TabularDataset(
            path=os.path.join(data_path, train_file), format='tsv',
            fields=train_fields
        )

        self.eval, self.test = data.TabularDataset.splits(
            path=data_path, validation=eval_file, test=test_file, format='tsv',
            fields=eval_fields)

        # build vocab
        text.build_vocab(self.train, self.eval, self.test)
        # text.build_vocab(self.train)
        self.train_vocab = text.vocab

        # load pretrain embedding
        vectors = vocab.Vectors(pretrain_ebd_file, data_path)
        self.train_vocab.load_vectors(vectors)

        # build batch iter
        self.train_iter = data.Iterator(
            self.train, sort_key=lambda x: len(x.wrong_answer),
            batch_size=train_batch_size, device=device,
            sort_within_batch=True, repeat=False, shuffle=shuffle
        )

        self.eval_iter, self.test_iter = data.Iterator.splits(
            (self.eval, self.test), sort_key=lambda x: len(x.answer),
            batch_sizes=(eval_batch_size, eval_batch_size), device=device,
            sort_within_batch=False, repeat=False, shuffle=False
        )

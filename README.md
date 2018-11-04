# Pytorch NLP Template Project

## Introduction
- This project is inspired by the project [pytorch-template](https://github.com/victoresque/pytorch-template) by [victoresque](https://github.com/victoresque)
- I made a small change in the data load module -- using torchtext instead of DataLoader, which is a very powerful library that solves the preprocessing of text very well.
- I finished a demo project about the Answer Selection dataset -- [WikiQA](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/). The model is according to the paper: [LSTM-BASED DEEP LEARNING MODELS FOR NONFACTOID ANSWER SELECTION](https://arxiv.org/pdf/1511.04108.pdf)
- See more details [here](https://github.com/victoresque/pytorch-template)

## Run Demo project
- Run download.sh to download the data file and pretrain word embedding
- Run train.sh and test.sh for training and testing

## TODOs
- Enable tensorboardX
- Design a better BaseNlpDataLoader

## Appendix
- [A good torchtext tutorial](http://anie.me/On-Torchtext/)
- [Torchtext documentation](https://torchtext.readthedocs.io/en/latest/)


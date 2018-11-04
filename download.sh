#! /bin/bash

set -e
set -x

mdir data

# download wikiqa data
wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip
unzip WikiQACorpus.zip
ln -s WikiQACorpus/WikiQA-*.tsv ./data/
rm WikiQACorpus.zip

# preprocesse data
python preprocessed.py ./data/WikiQA-train.tsv ./data/WikiQA-train-preprocessed.tsv

# download glove.840B.300d.
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
ln -s glove.840B.300d.txt ./data/glove.840B.300d.txt
rm glove.840B.300d.zip

echo "Done"

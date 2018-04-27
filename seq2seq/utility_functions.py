import sys
import numpy as np
sys.path.append('/home/tanvi/UMass/spring_2018/advanced-NLP/Project/TheStyleTransferProject')
from datautils import *
import os
from collections import defaultdict
import time
import math


"""
Always ensure that PAD = 0 and START = 1. word2num and num2word already take care of this.
"""

# function for creating vocab
# take path (can either be a document or a directory of documents)
# as input and create vocab from all documents present in that path (save a list of all words + UNK)
def create_vocab(path, min_count=1):
    print("Creating Vocab...")
    vocab = defaultdict(int)
    for root, direc, files in os.walk(path):
        if len(files) != 0:
            for fi in files:
                path = os.path.join(root, fi)
                sentences = document_tokenize(path)
                for sent in sentences:
                    for word in sent:
                        vocab[word] += 1
    final_vocab = [word for word, count in vocab.items() if count >= min_count]
    final_vocab += ['UNK']
    print('UNK' in final_vocab)
    print("Total vocab count, including UNK: {}".format(len(final_vocab)))
    final_vocab = '\n'.join(final_vocab)
    with open(VOCAB_PATH, 'w') as f:
        f.write(final_vocab)


# function for loading vocab
# load vocab - read the file
# return wordToNum and numToWord
def load_vocab():
    word2num = {"PAD": 0, "SOS": 1}
    with open(VOCAB_PATH, 'r') as f:
        words = f.read().split('\n')
    j = 2
    for w in words:
        word2num[w] = j
        j += 1

    num2word = {num: word for word, num in word2num.items()}
    return word2num, num2word


# function for converting batch of sentences to index vectors
# since the model is an autoencoder, output will be identical to input
# will need wordToNum
def convert_to_vector(batch, word2num):
    """
    :param batch: list of lists of words that form a single batch
    :return: list of lists of words, with words replaced with their ids
    """
    vectors = [word2num[word] if word in word2num else word2num['UNK'] for word in batch]
    return vectors


# function for selecting batch
# create a batch iteratively and convert sentences into index vectors (call the desired function)
def minibatches(data, word2num):
    """
    :param data: The input data that has to be broken into batches
    :param batch_size: desired batch size
    :param word2num: dictionary mapping words to their indices
    :return: encoder input, decoder input and decoder output
    """
    for line in data:
        enc_batch = convert_to_vector(line, word2num)
        sos_line = ["SOS"] + line
        dec_batch = convert_to_vector(sos_line, word2num)
        yield enc_batch, dec_batch, enc_batch


# function for converting batch of index vectors to sentences
# will need numToWord
def convert_to_sentence(vectors, num2word):
    sentences = [[num2word[num] for num in vec] for vec in vectors]
    return sentences


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

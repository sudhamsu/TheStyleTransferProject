import sys
import numpy as np
sys.path.append('../')
from datautils import *
import os
from collections import defaultdict
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


VOCAB_PATH = 'vocab.txt'


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('output/loss_curve.png')
    plt.close()

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
def minibatches(data, word2num, max_length=10):
    """
    :param data: The input data that has to be broken into batches
    :param batch_size: desired batch size
    :param word2num: dictionary mapping words to their indices
    :param max_length
    :return: encoder input, decoder input and decoder output
    """
    for a, line in data:
        batch = convert_to_vector(line, word2num)[:max_length]
        yield a, batch


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


class DataLoader(object):
    def __init__(self, data, word2num, batch_size, max_length):
        self.max_length = max_length
        self.word2num = word2num
        self.batch_size = batch_size
        self.data = data
        self.num_authors = len(data)
        self.data_lengths = [len(author_data) for author_data in data]
        self.authors_max_batches = [int(self.data_lengths[a]/self.batch_size) for a in range(self.num_authors)]
        self.authors_current_batch = [0] * self.num_authors

    def reset_author_batch_progress(self, a):
        self.authors_current_batch[a] = 0

    def get_next_training_batch(self):
        a = np.random.choice(range(self.num_authors))
        batch = self.vectorize(self.data[a][self.authors_current_batch[a] * self.batch_size:(self.authors_current_batch[a] + 1) * self.batch_size])

        self.authors_current_batch[a] += 1
        if self.authors_current_batch[a] >= self.authors_max_batches[a]:
            self.reset_author_batch_progress(a)

        return a, batch

    def vectorize(self, lines):
        batch = np.zeros([self.batch_size, self.max_length])
        for l, line in enumerate(lines):
            batch[l, min(len(line), self.max_length)] = np.array(convert_to_vector(line[:self.max_length], self.word2num))
        return batch
import sys
import numpy as np
sys.path.append('/home/tanvi/UMass/spring_2018/advanced-NLP/Project/TheStyleTransferProject')
from datautils import *
import os
from collections import defaultdict

"""
Always ensure that PAD = 0. word2num and num2word already take care of this and 0 is assigned 
to the word PAD
"""

# function for creating vocab
# take path (can either be a document or a directory of documents)
# as input and create vocab from all documents present in that path (save a list of all words + UNK)
def create_vocab(path, min_count=1):
    print("Creating Vocab...")
    vocab = defaultdict(int)
    for root, direc, files in os.walk(path):
        if files != []:
            for fi in files:
                path = os.path.join(root, fi)
                sentences = document_tokenize(path)
                for sent in sentences:
                    for word in sent:
                        vocab[word] += 1
    final_vocab = [word for word, count in vocab.items() if count >= min_count]
    final_vocab += ['UNK']
    print("Total vocab count, including UNK: {}".format(len(final_vocab)))
    final_vocab = '\n'.join(final_vocab)
    with open(VOCAB_PATH, 'w') as f:
        f.write(final_vocab)


# function for loading vocab
# load vocab - read the file
# return wordToNum and numToWord
def load_vocab():
    word2num = {"PAD":0}
    with open(VOCAB_PATH, 'r') as f:
        words = f.readlines()
    j = 1
    for w in enumerate(words):
        word2num[w] = j
        j += 1

    num2word = {num: word for word, num in word2num.items()}
    return word2num, num2word


# function for converting batch of sentences to index vectors
# since the model is an autoencoder, output will be identical to input
# will need wordToNum
def convert_to_vector(batch, word2num, batch_size):
    """
    :param batch: list of lists of words that form a single batch
    :return: list of lists of words, with words replaced with their ids
    """
    vectors = np.zeros(shape=[batch_size, max([len(line) for line in batch])], dtype=np.float32)
    for i, line in enumerate(batch):
        vectors[i][0:len(line)] = [word2num[word] for word in line]
    return vectors


# function for selecting batch
# create a batch iteratively and convert sentences into index vectors (call the desired function)
def minibatches(data, batch_size, word2num):
    """
    :param data: The input data that has to be broken into batches
    :param batch_size: desired batch size
    :param word2num: dictionary mapping words to their indices
    :return: input and output batches
    """
    x_batch = []
    for line in data:
        if len(x_batch) == batch_size:
            batch = convert_to_vector(x_batch, word2num, batch_size)
            x_batch = []
            yield batch, batch

        x_batch += line

    if len(x_batch) != 0:
        batch = convert_to_vector(x_batch, word2num, batch_size)
        yield batch, batch


# function for converting batch of index vectors to sentences
# will need numToWord
def convert_to_sentence(vectors, num2word):
    sentences = [[num2word[num] for num in vec] for vec in vectors]
    return sentences

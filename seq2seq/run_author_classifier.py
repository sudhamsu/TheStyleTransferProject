import numpy as np
import AuthorClassifier as ac
import utility_functions as uf
import datautils as du
import os
import datetime


EMBEDDING_DIM = 256
HIDDEN_DIM = 256
MAX_LENGTH=20
SENTS_PER_AUTHOR = 3500
LR = 1e-4
REG = 1e-4
EPOCHS = 10
PRINT_EVERY = 20


# GENERATE SAVE DIRECTORY PATH
timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
SAVE_DIR = 'output/' + timestamp + '/author_classifier'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

LOAD_DIR = SAVE_DIR  # TODO


authors = ["../Gutenberg/Fantasy/Howard_Pyle.txt", "../Gutenberg/Fantasy/William_Morris.txt"]

print('Loading data... ', end='')
word2num, num2word = uf.load_vocab()

data = [(a, sent)
        for a, author in enumerate(authors)
        for sent in np.random.choice(du.document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
                                     SENTS_PER_AUTHOR, replace=False)]
print('Done!\nTotal number of sentences:', len(data))

model = ac.train(data, word2num, len(authors), len(word2num),
                 embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                 epochs=EPOCHS, lr=LR, weight_decay=REG,
                 print_every=PRINT_EVERY, save_dir=SAVE_DIR)
ac.test(model, data, word2num)

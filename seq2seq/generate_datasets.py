import numpy as np
import datautils as du
import pickle
import os


MAX_LENGTH = 20
TRAIN_SENTS_PER_AUTHOR = 3300
TEST_SENTS_PER_AUTHOR = 200

authors = ["../Gutenberg/Fantasy/Howard_Pyle.txt", "../Gutenberg/Fantasy/William_Morris.txt"]
sents_per_author = TRAIN_SENTS_PER_AUTHOR + TEST_SENTS_PER_AUTHOR
data = [(a, sent)
        for a, author in enumerate(authors)
        for sent in np.random.choice(du.document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
                                     sents_per_author, replace=False)]

train_data = []
test_data = []
for a in range(len(authors)):
    start = a * sents_per_author
    test_start = a * sents_per_author + TRAIN_SENTS_PER_AUTHOR
    end = (a + 1) * sents_per_author
    train_data += data[start:test_start]
    test_data += data[test_start:end]

save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pickle.dump(train_data, open(save_dir+'/train.pkl', 'wb'))
pickle.dump(test_data, open(save_dir+'/test.pkl', 'wb'))

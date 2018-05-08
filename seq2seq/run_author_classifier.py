import numpy as np
import AuthorClassifier as ac
import utility_functions as uf
import datautils as du
import os
import datetime
import pickle
import sys
import torch

torch.set_flush_denormal(True)

TEST_ONLY = False
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
MAX_LENGTH = 20
SENTS_PER_AUTHOR = 3500
LR = 1e-4
REG = 1e-4
EPOCHS = 10
PRINT_EVERY = 20


# GENERATE SAVE DIRECTORY PATH
if not TEST_ONLY:
    timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
    SAVE_DIR = 'output/' + timestamp + '/author_classifier'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

LOAD_DIR = SAVE_DIR
# LOAD_DIR = '../../from_cloud/20180506_011129_author_clf_hp_wm/author_classifier'
# LOAD_DIR = '../../from_cloud/20180506_175829_author_clf_wm_lfb/author_classifier'
# RESULTS_LOAD_DIR = '../../Results/style_transfer_hp_wm_attn_0.5tf'
# RESULTS_LOAD_DIR = '../../Results/style_transfer_hp_wm_attn_tf'
# RESULTS_LOAD_DIR = '../../Results/style_transfer_hp_wm_noattn_0.5tf'
# RESULTS_LOAD_DIR = '../../Results/style_transfer_hp_wm_noattn_tf'
# RESULTS_LOAD_DIR = '../../Results/style_transfer_wm_lfb_attn_tf'

authors = ["../Gutenberg/Fantasy/Howard_Pyle.txt", "../Gutenberg/Fantasy/William_Morris.txt"]
TRAIN_PKL = 'data/fantasy_hp_wm_train.pkl'
TEST_PKL = 'data/fantasy_hp_wm_test.pkl'
# authors = ["../Gutenberg/Fantasy/William_Morris.txt", "../Gutenberg/Fantasy/Lyman_Frank_Baum.txt"]
# TRAIN_PKL = 'data/fantasy_wm_lfb_train.pkl'
# TEST_PKL = 'data/fantasy_wm_lfb_test.pkl'

print('Loading data... ', end='')
word2num, num2word = uf.load_vocab()

if not TEST_ONLY:
    # data = [(a, sent)
    #         for a, author in enumerate(authors)
    #         for sent in np.random.choice(du.document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
    #                                      SENTS_PER_AUTHOR, replace=False)]
    data = pickle.load(open(TRAIN_PKL, 'rb'))
    print('Done!\nTotal number of sentences:', len(data))

    model = ac.train(data, word2num, len(authors), len(word2num),
                     embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                     epochs=EPOCHS, lr=LR, weight_decay=REG,
                     print_every=PRINT_EVERY, save_dir=SAVE_DIR)
else:
    model = ac.AuthorClassifier(len(authors), len(word2num), EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(LOAD_DIR+'/final_model.pth'))

# test_data = data
test_data = pickle.load(open(TEST_PKL, 'rb'))
print('Done!\nSize of test set:', len(test_data))
test_acc = ac.test(model, test_data, word2num)

# For classifying style-transferred sentences
# if os.path.exists(RESULTS_LOAD_DIR + '/target_predictions.pkl'):
#     translated_data = pickle.load(open(RESULTS_LOAD_DIR + '/target_predictions.pkl', 'rb'))
#     print('Size of translated set:', len(translated_data))
#     ac.test_translations(model, translated_data, word2num, RESULTS_LOAD_DIR, test_acc)

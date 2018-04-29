from utility_functions import *
from datautils import *
import StyleTransferModel
import StyleTransferTrainer
import os
import torch
import datetime
import pickle


MAX_LENGTH = 20
HIDDEN_SIZE = 256
BATCH_SIZE = 1  # TODO: current code implementation is valid ONLY for batch size 1
MIN_COUNT = 5
SENTS_PER_AUTHOR = 3500  # author 2 below has only 3500+epsilon 20-length sentences
TEST_SENTS_PER_AUTHOR = 10
PRINT_EVERY = 20
LR = 1e-2
EPOCHS = 7
CHECKPOINT_INTERVAL = 2  # number of epochs after which to save checkpoint


# GENERATE SAVE DIRECTORY PATH
timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
if not os.path.exists('output'):
    os.makedirs('output')
SAVE_DIR = 'output/' + timestamp
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

LOAD_DIR = SAVE_DIR  # TODO


authors = ["../Gutenberg/Fantasy/Howard_Pyle.txt", "../Gutenberg/Fantasy/William_Morris.txt"]

print('Loading data... ', end='')

if not os.path.exists('vocab.txt'):
    create_vocab('../Gutenberg', min_count=MIN_COUNT)

word2num, num2word = load_vocab()


# TRAINING STYLE TRANSFER MODEL
# data = [(a, sent)
#         for a, author in enumerate(authors)
#         for sent in np.random.choice(document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
#                                      SENTS_PER_AUTHOR, replace=False)]
data = pickle.load(open('data/train.pkl', 'rb'))
print('Done!\nTotal number of sentences:', len(data))
# data = [document_tokenize(author, tokenize_words=True) for author in authors]
# dlo = DataLoader(data, word2num, BATCH_SIZE, MAX_LENGTH)

encoder = StyleTransferModel.EncoderRNN(len(word2num), HIDDEN_SIZE)
decoders = [StyleTransferModel.AttnDecoderRNN(HIDDEN_SIZE, len(word2num), MAX_LENGTH, dropout_p=0) for _ in authors]

StyleTransferTrainer.train_iters(word2num, data, encoder, decoders, MAX_LENGTH,
                                 epochs=EPOCHS, learning_rate=LR,
                                 print_every=PRINT_EVERY, save_dir=SAVE_DIR, checkpoint_interval=CHECKPOINT_INTERVAL)


# TESTING STYLE TRANSFER MODEL
# test_data = data
# test_data = [(a, sent)
#              for a, author in enumerate(authors)
#              for sent in np.random.choice(document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
#                                           TEST_SENTS_PER_AUTHOR, replace=False)]
test_data = pickle.load(open('data/test.pkl', 'rb'))
test_data = np.random.choice(test_data, 20, replace=False)

encoder = StyleTransferModel.EncoderRNN(len(word2num), HIDDEN_SIZE)
decoders = [StyleTransferModel.AttnDecoderRNN(HIDDEN_SIZE, len(word2num), MAX_LENGTH, dropout_p=0) for _ in authors]

encoder.load_state_dict(torch.load(LOAD_DIR+'/final_encoder.pth'))
for i, d in enumerate(decoders):
    d.load_state_dict(torch.load(LOAD_DIR+'/final_decoder'+str(i)+'.pth'))

StyleTransferTrainer.test(len(authors), word2num, num2word, test_data, encoder, decoders, MAX_LENGTH, save_dir=SAVE_DIR)

from utility_functions import *
from datautils import *
import StyleTransferModel
import StyleTransferTrainer
from content_preservation_test import *
import os
import torch
import datetime
import pickle
import sys


MAX_LENGTH = 20
HIDDEN_SIZE = 256
BATCH_SIZE = 1  # TODO: current code implementation is valid ONLY for batch size 1
MIN_COUNT = 5
SENTS_PER_AUTHOR = 3300  # author 2 below has only 3500+epsilon 20-length sentences
TEST_SENTS_PER_AUTHOR = 200
PRINT_EVERY = 20
LR = 1e-2
EPOCHS = 5
CHECKPOINT_INTERVAL = 2  # number of epochs after which to save checkpoint


# GENERATE SAVE DIRECTORY PATH
timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
if not os.path.exists('output'):
    os.makedirs('output')
SAVE_DIR = 'output/' + timestamp
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


LOAD_DIR = SAVE_DIR

# authors = ["../Gutenberg/Fantasy/Howard_Pyle.txt", "../Gutenberg/Fantasy/William_Morris.txt"]
authors = ["../Gutenberg/Fantasy/William_Morris.txt", "../Gutenberg/Fantasy/Lyman_Frank_Baum.txt"]

print('Loading data... ', end='')

if not os.path.exists('vocab.txt'):
    create_vocab('../Gutenberg', min_count=MIN_COUNT)

word2num, num2word = load_vocab()

# TRAINING STYLE TRANSFER MODEL
# data = [(a, sent)
#         for a, author in enumerate(authors)
#         for sent in np.random.choice(document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
#                                      SENTS_PER_AUTHOR, replace=False)]

# LOAD TRAINING SET FROM PICKLED FILE
# data = pickle.load(open('data/fantasy_hp_wm_train.pkl', 'rb'))
data = pickle.load(open('data/fantasy_wm_lfb_train.pkl', 'rb'))

print('Done!\nTotal number of sentences:', len(data))
sys.stdout.flush()

encoder = StyleTransferModel.Encoder(len(word2num), HIDDEN_SIZE)
decoders = [StyleTransferModel.AttentionDecoder(HIDDEN_SIZE, len(word2num), MAX_LENGTH, dropout_p=0) for _ in authors]
# decoders = [StyleTransferModel.Decoder(HIDDEN_SIZE, len(word2num)) for _ in authors]

StyleTransferTrainer.train_iters(word2num, data, encoder, decoders, MAX_LENGTH,
                                 epochs=EPOCHS, learning_rate=LR,
                                 print_every=PRINT_EVERY, save_dir=SAVE_DIR, checkpoint_interval=CHECKPOINT_INTERVAL)


# TESTING STYLE TRANSFER MODEL
# test_data = data  # when we want training set = test set
# test_data = [(a, sent)
#              for a, author in enumerate(authors)
#              for sent in np.random.choice(document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
#                                           TEST_SENTS_PER_AUTHOR, replace=False)]

# LOAD TEST SET FROM PICKLED FILE
# loaded_test_data = pickle.load(open('data/fantasy_hp_wm_test.pkl', 'rb'))
loaded_test_data = pickle.load(open('data/fantasy_wm_lfb_test.pkl', 'rb'))
chosen_indices = np.random.choice(np.arange(len(loaded_test_data)), len(authors)*TEST_SENTS_PER_AUTHOR, replace=False).tolist()
test_data = []
for i in chosen_indices:
    test_data.append(loaded_test_data[i])
del loaded_test_data

encoder = StyleTransferModel.Encoder(len(word2num), HIDDEN_SIZE)
decoders = [StyleTransferModel.AttentionDecoder(HIDDEN_SIZE, len(word2num), MAX_LENGTH, dropout_p=0) for _ in authors]
# decoders = [StyleTransferModel.Decoder(HIDDEN_SIZE, len(word2num)) for _ in authors]

encoder.load_state_dict(torch.load(LOAD_DIR+'/final_encoder.pth'))
for i, d in enumerate(decoders):
    d.load_state_dict(torch.load(LOAD_DIR+'/final_decoder'+str(i)+'.pth'))

predictions = StyleTransferTrainer.test(word2num, num2word, test_data, encoder, decoders, MAX_LENGTH)
cosine_similarity_vector = compute_content_preservation(predictions)

about = "The list of authors used: {}\n".format(authors)
print(about, file=open(LOAD_DIR + '/test_predictions.txt', 'a+'))
sys.stdout.flush()
target_predictions = []
for i in range(len(predictions)):
    a = test_data[i][0]                 # index of input author
    b = (a + 1) % len(authors)          # index of target author
    print_string = '\n' + 'Original (Author {}): {}'.format(a, predictions[i][0])
    print_string += '\n' + 'Transferred (Author {}): {}'.format(b, predictions[i][1])
    print_string += '\n' + 'Cosine Similarity: {}'.format(cosine_similarity_vector[i])
    print(print_string)
    print(print_string, file=open(LOAD_DIR + '/test_predictions.txt', 'a+'))
    sys.stdout.flush()
    target_predictions.append((b, predictions[i][1]))

with open(LOAD_DIR + "/cosine_similarity.pkl", 'wb') as f:
    pickle.dump(cosine_similarity_vector, f)

with open(LOAD_DIR + "/target_predictions.pkl", "wb") as f:
    pickle.dump(target_predictions, f)

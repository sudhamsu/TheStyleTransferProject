from utility_functions import *
from datautils import *
import StyleTransferModel
import StyleTransferTrainer
from content_preservation_test import *
import os
import torch
import datetime


MAX_LENGTH = 20
HIDDEN_SIZE = 256
BATCH_SIZE = 1  # TODO: current code implementation is valid ONLY for batch size 1
MIN_COUNT = 5
SENTS_PER_AUTHOR = 3500  # author 2 below has only 3500+epsilon 20-length sentences
TEST_SENTS_PER_AUTHOR = 10
PRINT_EVERY = 20
LR = 1e-2
EPOCHS = 5
CHECKPOINT_INTERVAL = 2  # number of epochs after which to save checkpoint


# GENERATE SAVE DIRECTORY PATH
#timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
#if not os.path.exists('output'):
#    os.makedirs('output')
#SAVE_DIR = 'output/' + timestamp
#if not os.path.exists(SAVE_DIR):
#    os.makedirs(SAVE_DIR)

LOAD_DIR = 'output/20180429_135307_0.5_teacher_enforcing'
#SAVE_DIR  # TODO


authors = ["../Gutenberg/Fantasy/Howard_Pyle.txt", "../Gutenberg/Fantasy/William_Morris.txt"]

print('Loading data... ', end='')

if not os.path.exists('vocab.txt'):
    create_vocab('../Gutenberg', min_count=MIN_COUNT)

word2num, num2word = load_vocab()

#data = [(a, sent)
#        for a, author in enumerate(authors)
#        for sent in np.random.choice(document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
#                                    SENTS_PER_AUTHOR, replace=False)]
#print('Done!\nTotal number of sentences:', len(data))
# data = [document_tokenize(author, tokenize_words=True) for author in authors]
# dlo = DataLoader(data, word2num, BATCH_SIZE, MAX_LENGTH)

#encoder = StyleTransferModel.EncoderRNN(len(word2num), HIDDEN_SIZE)
#decoders = [StyleTransferModel.AttnDecoderRNN(HIDDEN_SIZE, len(word2num), MAX_LENGTH, dropout_p=0) for _ in authors]

#StyleTransferTrainer.train_iters(word2num, data, encoder, decoders, MAX_LENGTH,
#                                 epochs=EPOCHS, learning_rate=LR,
#                                 print_every=PRINT_EVERY, save_dir=SAVE_DIR, checkpoint_interval=CHECKPOINT_INTERVAL)


test_data = [(a, sent)
             for a, author in enumerate(authors)
             for sent in np.random.choice(document_tokenize(author, max_length=MAX_LENGTH, tokenize_words=True),
                                          TEST_SENTS_PER_AUTHOR, replace=False)]
# test_data = data

encoder = StyleTransferModel.EncoderRNN(len(word2num), HIDDEN_SIZE)
decoders = [StyleTransferModel.AttnDecoderRNN(HIDDEN_SIZE, len(word2num), MAX_LENGTH, dropout_p=0) for _ in authors]

encoder.load_state_dict(torch.load(LOAD_DIR+'/encoder_after_epoch_3.pth'))
for i, d in enumerate(decoders):
    d.load_state_dict(torch.load(LOAD_DIR+'/decoder'+str(i)+'_after_epoch_3.pth'))

predictions = StyleTransferTrainer.test(len(authors), word2num, num2word, test_data, encoder, decoders, MAX_LENGTH, save_dir=LOAD_DIR)
cosine_similarity_vector = compute_content_preservation(predictions)
for i in range(len(predictions)):
    a = test_data[i][0]
    b = (a + 1) % len(authors)
    print("\nOriginal (Author {}): {}".format(a, predictions[i][0]))
    print("Transferred (Author {}): {}".format(b, predictions[i][1]))
    print("Cosine Similarity: {}".format(cosine_similarity_vector[i]))

#print("Cosine Similarity of all examples: {}".format(cosine_similarity_vector))
with open(LOAD_DIR + "/cosine_similarity_with_tf.pkl", 'wb') as f:
    pickle.dump(cosine_similarity_vector, f)


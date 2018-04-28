from utility_functions import *
from datautils import *
import StyleTransferModel
import StyleTransferTrainer


MAX_LENGTH = 10
HIDDEN_SIZE = 64
BATCH_SIZE = 1  # TODO: current code implementation is valid ONLY for batch size 1
MIN_COUNT = 5


authors = ["../Gutenberg/Fantasy/Howard_Pyle.txt", "../Gutenberg/Fantasy/William_Morris.txt"]


create_vocab('../Gutenberg', min_count=MIN_COUNT)
word2num, num2word = load_vocab()
data = [(a, sent)
        for a, author in enumerate(authors)
        for sent in document_tokenize(author, tokenize_words=True)[:5000]]  # TODO: taking only first 5000 in each author
np.random.shuffle(data)
print('Total number of sentences:', len(data))
# data = [document_tokenize(author, tokenize_words=True) for author in authors]
# dlo = DataLoader(data, word2num, BATCH_SIZE, MAX_LENGTH)

encoder = StyleTransferModel.EncoderRNN(len(word2num), HIDDEN_SIZE)
decoders = [StyleTransferModel.AttnDecoderRNN(HIDDEN_SIZE, len(word2num), MAX_LENGTH) for _ in authors]

StyleTransferTrainer.train_iters(word2num, data, encoder, decoders, MAX_LENGTH, epochs=1, print_every=20)

# TODO: test
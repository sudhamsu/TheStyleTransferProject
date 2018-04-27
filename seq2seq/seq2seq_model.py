from __future__ import unicode_literals, print_function, division
import sys
sys.path.append('/home/tanvi/UMass/spring_2018/advanced-NLP/Project/TheStyleTransferProject')
from utility_functions import *
from datautils import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cpu")


"""
Lets start by experimenting with only two authors, and setting up a system to transfer styles 
between author 1 and author 2. 

http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html <-- code adapted from here
"""

author1 = "../Gutenberg/Fantasy/Howard_Pyle.txt"
author2 = "../Gutenberg/Fantasy/William_Morris.txt"
batch_size = 1
min_count = 5
SOS = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: vocab size
        :param hidden_size: number of hidden units
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True
    #if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == 0:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def trainIters(data, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    #plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    max_len = max([len(line) for line in data])

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [torch.tensor(enc_input, dtype=torch.long, device=device).view(-1, 1)
                      for enc_input, dec_input, dec_output in minibatches(data, word2num)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_len)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        #if iter % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0


if __name__ == '__main__':

    create_vocab('../Gutenberg', min_count=min_count)
    word2num, num2word = load_vocab()
    data1 = document_tokenize(author1, tokenize_words=True)
    data2 = document_tokenize(author2, tokenize_words=True)

    hidden_size = 256
    encoder1 = EncoderRNN(len(word2num), hidden_size).to(device)
    attn_decoder1 = DecoderRNN(hidden_size, len(word2num)).to(device)

    trainIters(data1, encoder1, attn_decoder1, 75000, print_every=5000)

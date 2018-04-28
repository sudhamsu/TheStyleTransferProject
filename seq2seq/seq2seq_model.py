from __future__ import unicode_literals, print_function, division
import sys
sys.path.append('../')
from utility_functions import *
from datautils import *
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# device = torch.device("cpu")


"""
Lets start by experimenting with only two authors, and setting up a system to transfer styles 
between author 1 and author 2. 

http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html <-- code adapted from here
"""

MAX_LENGTH = 10

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
        return torch.zeros(1, 1, self.hidden_size)#, device=device)


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
        return torch.zeros(1, 1, self.hidden_size)#, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)#, device=device)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)#, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]])#, device=device)

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



def trainIters(data, encoder, decoder, epochs, print_every=1000, plot_every=100, learning_rate=0.01):
    print('Inside trainIters...')
    start = time.time()
    #plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # max_len = max([len(line) for line in data])
    max_len = MAX_LENGTH

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for e in range(epochs):
        iter = 0
        for batch in minibatches(data, word2num):
            print('iter:', iter)
            training_pair = torch.LongTensor(batch).view(-1, 1)#, device=device).view(-1, 1)
            input_tensor = training_pair
            target_tensor = training_pair

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, max_len)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('loss:',print_loss_avg)
                # print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                #                              iter, iter / n_iters * 100, print_loss_avg))

            #if iter % plot_every == 0:
            #    plot_loss_avg = plot_loss_total / plot_every
            #    plot_losses.append(plot_loss_avg)
            #    plot_loss_total = 0

            iter += 1


if __name__ == '__main__':

    create_vocab('../Gutenberg', min_count=min_count)
    word2num, num2word = load_vocab()
    data1 = document_tokenize(author1, tokenize_words=True)
    data2 = document_tokenize(author2, tokenize_words=True)

    hidden_size = 256
    encoder1 = EncoderRNN(len(word2num), hidden_size)#.to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, len(word2num))#.to(device)

    trainIters(data1, encoder1, attn_decoder1, 5, print_every=1)

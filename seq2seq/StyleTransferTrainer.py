from utility_functions import *
import torch
import torch.nn as nn
from torch import optim
import datetime


SOS = 1


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


def train_iters(word2num, data, encoder, decoders, max_length, epochs=5, print_every=1000, learning_rate=0.01):
    start = time.time()
    loss_total = 0  # Reset every print_every
    losses = []
    # max_len = max([len(line) for line in data])

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizers = [optim.SGD(d.parameters(), lr=learning_rate) for d in decoders]
    criterion = nn.NLLLoss()

    for e in range(epochs):
        itr = 1
        for a, batch in minibatches(data, word2num, max_length=max_length):
            input_tensor = torch.LongTensor(batch).view(-1, 1)

            loss = train(input_tensor, input_tensor, encoder, decoders[a],
                         encoder_optimizer, decoder_optimizers[a], criterion, max_length)
            loss_total += loss

            if itr % print_every == 0:
                print_loss_avg = loss_total / print_every
                loss_total = 0
                losses.append(print_loss_avg)
                print('{:%H:%M:%S} loss:{}'.format(datetime.datetime.now(), print_loss_avg))
                # print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                #                              iter, iter / n_iters * 100, print_loss_avg))

            itr += 1

    torch.save(encoder.state_dict(), 'output/encoder.pth')
    for i, d in enumerate(decoders):
        torch.save(d.state_dict(), 'output/decoder'+str(i)+'.pth')
    torch.save(encoder_optimizer.state_dict(), 'output/encoder_optimizer.pth')
    for i, d in enumerate(decoder_optimizers):
        torch.save(d.state_dict(), 'output/decoder_optimizer' + str(i) + '.pth')
    showPlot(losses)
import utility_functions as uf
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import time


EOS = 0
SOS = 1


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, max_length):
    encoder_hidden = encoder.init_hidden_state()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    for e in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[e], encoder_hidden)
        encoder_outputs[e] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]])

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True
    #if np.random.rand() < 0.5:
    #    use_teacher_forcing = False

    if use_teacher_forcing:
        for d in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += loss_function(decoder_output, target_tensor[d])
            decoder_input = target_tensor[d]
    else:
        for d in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += loss_function(decoder_output, target_tensor[d])
            if decoder_input.item() == EOS:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(word2num, data, encoder, decoders, max_length, epochs=5, print_every=1000, learning_rate=1e-2,
                save_dir='output', checkpoint_interval=1):

    print('Setting up trainer... ', end='')
    encoder.train()
    for d in decoders:
        d.train()

    loss_total = 0  # Reset every print_every
    losses = []
    # max_len = max([len(line) for line in data])

    encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate)
    decoder_optimizers = [optim.Adagrad(d.parameters(), lr=learning_rate) for d in decoders]
    criterion = nn.NLLLoss()

    iters_per_epoch = len(data)

    print('Done!\nTraining...')
    start = time.time()
    for e in range(epochs):
        itr = 0
        np.random.shuffle(data)
        for a, batch in uf.minibatches(data, word2num, max_length=max_length):
            input_tensor = torch.LongTensor(batch).view(-1, 1)

            loss = train(input_tensor, input_tensor, encoder, decoders[a],
                         encoder_optimizer, decoder_optimizers[a], criterion, max_length)
            loss_total += loss

            if (itr + 1) % print_every == 0:
                print_loss_avg = loss_total / print_every
                loss_total = 0
                losses.append(print_loss_avg)
                print('{:3}%  t: {:21}  e: {:3}  i: {:4}  loss: {:.8}'.format(
                    int((itr + 1 + iters_per_epoch * e) / (iters_per_epoch * epochs) * 100),
                    uf.timeSince(start, (itr + 1 + iters_per_epoch * e) / (iters_per_epoch * epochs)),
                    str(e + 1), str(itr + 1), print_loss_avg))

            itr += 1

        if (e + 1) % checkpoint_interval == 0:  # save checkpoint after every checkpoint_interval epochs
            torch.save(encoder.state_dict(), save_dir + '/encoder_after_epoch_'+str(e)+'.pth')
            for i, d in enumerate(decoders):
                torch.save(d.state_dict(), save_dir + '/decoder' + str(i) + '_after_epoch_'+str(e)+'.pth')
            torch.save(encoder_optimizer.state_dict(), save_dir + '/encoder_optimizer_after_epoch_'+str(e)+'.pth')
            for i, d in enumerate(decoder_optimizers):
                torch.save(d.state_dict(), save_dir + '/decoder_optimizer' + str(i) + '_after_epoch_'+str(e)+'.pth')
            uf.save_plot(save_dir + '/loss_curve_after_epoch_' + str(e) + '.png', losses, print_every)
            np.save(save_dir+'/losses_after_epoch_'+str(e), np.array(losses))

    torch.save(encoder.state_dict(), save_dir+'/final_encoder.pth')
    for i, d in enumerate(decoders):
        torch.save(d.state_dict(), save_dir+'/final_decoder'+str(i)+'.pth')
    torch.save(encoder_optimizer.state_dict(), save_dir+'/final_encoder_optimizer.pth')
    for i, d in enumerate(decoder_optimizers):
        torch.save(d.state_dict(), save_dir+'/final_decoder_optimizer' + str(i) + '.pth')
    uf.save_plot(save_dir + '/final_loss_curve.png', losses, print_every)
    np.save(save_dir + '/final_losses', losses)


def evaluate(input_tensor, num2word, encoder, decoder, max_length):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden_state()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for e in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[e], encoder_hidden)
            encoder_outputs[e] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS]])

        decoder_hidden = encoder_hidden

        decoded_words = []

        for d in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS:
                decoded_words.append(num2word[EOS])
                break
            else:
                decoded_words.append(num2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def test(word2num, num2word, data, encoder, decoders, max_length):
    encoder.eval()
    for d in decoders:
        d.eval()

    source_prediction_pairs = []
    for a, batch in uf.minibatches(data, word2num, max_length=max_length):
        input_tensor = torch.LongTensor(batch).view(-1, 1)
        output = evaluate(input_tensor, num2word, encoder, decoders[a], max_length)
        original = ' '.join([num2word[w] for w in batch])
        transferred = ' '.join(output)
        source_prediction_pairs.append((original, transferred))
    return source_prediction_pairs

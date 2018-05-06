import torch
import torch.nn as nn
import numpy as np
import utility_functions as uf
import time
import sys


# TODO NOTE: this code is valid only for batch size 1, other sizes would need padding


class AuthorClassifier(nn.Module):
    def __init__(self, num_authors, vocab_size, embedding_dim, hidden_dim):
        super(AuthorClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, num_authors)
        self.lstm_hidden_state = None

    def init_lstm_hidden_state(self, batch_size=1):
        # The dimensions are: (num_layers (?), minibatch_size, hidden_dim)
        return (torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def forward(self, batch):
        batch_embed = self.embedding(batch)
        batch_embed = batch_embed.view(batch.size(1), batch.size(0), -1)  # TODO make sure this is right
        lstm_output, self.lstm_hidden_state = self.lstm(batch_embed, self.lstm_hidden_state)
        output = self.linear(lstm_output[-1])
        return output


def train(data, word2num, num_authors, vocab_size, embedding_dim=64, hidden_dim=64,
          lr=1e-4, weight_decay=1e-4, epochs=50, print_every=1, save_dir='output'):
    model = AuthorClassifier(num_authors, vocab_size, embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()

    losses = []
    iters_per_epoch = len(data)

    model.train()
    start = time.time()
    for e in range(epochs):
        np.random.shuffle(data)
        total_loss_print_every = 0
        for i in range(iters_per_epoch):
            model.lstm_hidden_state = model.init_lstm_hidden_state()

            optimizer.zero_grad()

            a, line = data[i]
            line = torch.autograd.Variable(torch.LongTensor([uf.convert_to_vector(line, word2num)]))
            label = torch.autograd.Variable(torch.FloatTensor([[1 if i==a else 0 for i in range(num_authors)]]))
            output = model(line)
            loss = loss_function(output, label)
            total_loss_print_every += loss.item()
            loss.backward()
            optimizer.step()

            if (i+1) % print_every == 0:
                print_loss_avg = total_loss_print_every / print_every
                total_loss_print_every = 0
                losses.append(print_loss_avg)
                print('{:3}%  t: {:21}  e: {:3}  i: {:4}  loss: {:.8}'.format(
                    int((i+1 + iters_per_epoch * e) / (iters_per_epoch * epochs) * 100),
                    uf.timeSince(start, (i+1 + iters_per_epoch * e) / (iters_per_epoch * epochs)),
                    str(e + 1), str(i + 1), print_loss_avg))
                sys.stdout.flush()

    torch.save(model.state_dict(), save_dir + '/final_model.pth')
    torch.save(optimizer.state_dict(), save_dir + '/final_optimizer.pth')
    uf.save_plot(save_dir + '/final_loss_curve.png', losses, print_every)
    np.save(save_dir + '/final_losses', losses)

    return model


def test(model, data, word2num):
    predictions = predict(model, data, word2num)
    count_correct_preds = 0

    for i in range(len(data)):
        count_correct_preds += (predictions[i] == data[i][0])

    test_acc = count_correct_preds / len(data)
    print('Test accuracy: {:.4f}%'.format(test_acc*100))
    sys.stdout.flush()


def predict(model, data, word2num):
    model.eval()
    softmax = nn.Softmax(dim=1)
    predictions = []

    for i in range(len(data)):
        model.zero_grad()
        model.lstm_hidden_state = model.init_lstm_hidden_state()

        a, line = data[i]
        line = torch.autograd.Variable(torch.LongTensor([uf.convert_to_vector(line, word2num)]))
        output = model(line)
        output = softmax(output)
        _, pred = torch.max(output, 1)

        predictions.append(pred.item())

    return predictions

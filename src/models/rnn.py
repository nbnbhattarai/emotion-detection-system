'''
RNN models for emotion-detection
'''
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score

import mlflow

from data.make_dataset import load_emotion
from features.build_features import get_glove_features

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    '''
    LSTM encoder module
    '''

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hidden_size)

    def forward(self, inp, hidden_state):
        '''Forward propagate'''
        # print(inp.shape)
        # print(f'forward: {inp}')
        return self.lstm(inp.view((1, 1, -1)), hidden_state)

    def init_hidden(self):
        '''initialization for hidden layer'''
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))


class Decoder(nn.Module):
    '''
    Output module for encoder decoder
    '''

    def __init__(self, n_class, hidden_size):
        super().__init__()
        self.lin_layer = nn.Linear(hidden_size, 512)
        self.out_layer = nn.Linear(512, n_class)

    def forward(self, inp):
        '''Forward propagate'''
        out1 = torch.tanh(self.lin_layer(inp))
        return F.softmax(self.out_layer(out1).squeeze(), dim=0)


def train(x_i, y_i, encoder, decoder, enc_optim, dec_optim, criterion, device):
    '''Train for a single input,output (x_i, y_i)'''
    hidden, cell = encoder.init_hidden()
    hidden = hidden.to(device)
    cell = cell.to(device)

    enc_optim.zero_grad()
    dec_optim.zero_grad()

    x_i = torch.tensor(x_i, dtype=torch.float32, device=device)
    y_i = torch.tensor(y_i, dtype=torch.long, device=device).view(1)

    n_len = x_i.size(0)

    logger.info(f'Shape of X_i: {X_i.shape}, dtype: {X_i.dtype}')
    logger.info(f'Shape of y_i: {y_i.shape}, dtype: {y_i.dtype}')

    for e_i in range(n_len):
        _, (hidden, cell) = encoder(x_i[e_i], (hidden, cell))

    logger.info('Encoder process completed')
    logger.info(f'Hidden state shape: {hidden.shape}')

    decoder_output = decoder(hidden).squeeze()
    logger.info(
        f'Decoder output: {decoder_output}, shape: {decoder_output.shape}')
    logger.info(
        f'True target value, y_i: {y_i}, shape: {y_i.shape}, dtype: {y_i.dtype}')

    loss = criterion(decoder_output.view(1, 7), y_i)
    logger.info(f'Calculated loss: {loss}, {loss.item()}')

    loss.backward()

    enc_optim.step()
    dec_optim.step()

    return loss.item()


def train_iters(x_train, y_train, x_test, y_test, n_epochs, encoder, decoder, enc_optim, dec_optim, criterion, device):
    '''
    Run iterations on encoder-decoder
    '''
    for i in range(n_epochs):
        num_x = len(x_train)

        total_loss = 0
        for x_i in range(num_x):
            total_loss += train(x_train[x_i], y_train[x_i], encoder, decoder,
                                enc_optim, dec_optim, criterion, device)

        avg_loss = total_loss/num_x
        test_acc = get_accuracy(
            x_test, y_test, encoder, decoder)

        print(
            f'Epoch: {i}, Loss: {avg_loss:.5f}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')


def predict(X, encoder, decoder):
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32, device=device)
        X_len = X.size(0)

        hidden, cell = encoder.init_hidden()

        for ei in range(X_len):
            _, (hidden, cell) = encoder(X[ei], (hidden, cell))

        decoder_output = decoder(hidden).squeeze()

        return decoder_output


def get_accuracy(X, y, encoder, decoder):
    y_pred = [np.argmax(predict(Xi, encoder, decoder).cpu()) for Xi in X]
    accuracy = accuracy_score(y, y_pred)
    return accuracy


class RNNModel(object):
    def __init__(self, test_size):
        self.test_size = test_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available else 'cpu')
        self.encoder = Encoder(1024).to(self.device)
        self.decoder = Decoder(7, 1024).to(self.device)

        self.enc_optim = optim.Adam(self.encoder.parameters())
        self.dec_optim = optim.Adam(self.decoder.parameters())

        self.criterion = nn.NLLLoss()

    def fit(self, n_epochs):
        self.dataset = load_emotion()
        self.X, self.y = get_glove_features(self.dataset)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size)

        train_iters(self.x_train, self.y_train, self.x_test, self.y_test, n_epochs,
                    self.enc_optim, self.dec_optim, self.criterion, self.device)

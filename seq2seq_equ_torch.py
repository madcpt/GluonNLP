import collections
import io
import math

import torch
from torch import nn
from torch.nn import init

from data.Equation_torch import Equation
from utils import draw


class Encoder(nn.Module):
    def __init__(self, vocal_size, embed_dim, num_hiddens, 
                num_layers, device, *args):
        super(Encoder, self).__init__(*args)
        self.embedding = nn.Embedding(vocal_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, num_hiddens, num_layers)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.device = device

    def forward(self, inputs, state):
        '''
        input (seq_len, batch, input_size)
        state (num_layers * num_directions, batch, hidden_size)
        output (seq_len, batch, num_directions * hidden_size)
        '''
        embedding = self.embedding(inputs.t())
        return self.rnn(embedding, state)
    
    def begin_state(self, batch_size, *args, **kwargs):
        return torch.zeros((self.num_layers, batch_size, self.num_hiddens), device=self.device)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hiddens,
                num_layers, *args):
        super(Decoder, self).__init__(*args)
        self.embedding = nn.Embedding(vocal_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, num_hiddens, num_layers)
        self.dense = nn.Linear(num_hiddens, vocab_size)
    
    def forward(self, inputs, state):
        '''
        input (seq_len, batch, input_size)
        state (num_layers * num_directions, batch, hidden_size)
        output (seq_len, batch, num_directions * hidden_size)
        '''        
        x = self.embedding(inputs)
        output, state = self.rnn(x, state)
        Y = self.dense(output)
        return Y, state

    def begin_state(self, enc_state):
        return enc_state

        

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    batch_size = 100
    max_len = 99
    dataset = Equation(max_len, device=device)
    dataset.generate_train_test(0.2)

    vocal_size = len(dataset.idx2str)
    embed_dim = 20
    num_hiddens = 64
    num_epoch = 200

    encoder = Encoder(vocal_size, embed_dim, num_hiddens, 2, device).to(device)
    decoder = Decoder(vocal_size, embed_dim, num_hiddens, 2).to(device)
    # model = EncoderDecoder(encoder, decoder, dataset).to(device)
    
    for params in encoder.parameters():
        init.normal_(params, mean=0, std=1)
    
    loss = torch.nn.CrossEntropyLoss()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.1)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.1)

    display_data = []

    for epoch in range(num_epoch):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        total_loss = 0
        iter = dataset.generate_data_iter(batch_size)
        for x,y in iter:
            encoder_state = encoder.begin_state(batch_size=batch_size)
            _, encoder_state = encoder(x, encoder_state)
            if isinstance (encoder_state, tuple): # LSTM, encoder_state:(h, c)  
                encoder_state = (encoder_state[0].detach(), encoder_state[1].detach())
            else:   
                encoder_state = encoder_state.detach()

            decoder_state = decoder.begin_state(encoder_state)
            
            for i in range(y.shape[1]-1):
                y_t = y.t()[i]
                Y, decoder_state = decoder(torch.reshape(y_t, (1,*y_t.shape)), decoder_state)
                output = Y.argmax(-1)
                l = loss(Y[0], y.t()[i+1])
                total_loss += l
        if (epoch+1)%10==0:
            print(total_loss)
            draw(display_data)
        display_data.append(total_loss.item())
        total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        # if i % 50 == 0:
        #     print(i, end=', ')
        #     print(total_loss.asscalar())

    for x,y in dataset.generate_data_iter(1, 'train'):
        encoder_state = encoder.begin_state(batch_size=1)
        _, encoder_state = encoder(x, encoder_state)
        decoder_state = decoder.begin_state(encoder_state)
        str_equ = [dataset.idx2str[int(i.item())] for i in x[0]]
        str_result = []
        for i in range(y.shape[1]-1):
            y_i = y.t()[i]
            output, decoder_state = decoder(y_i.reshape((1,*y_i.shape)), decoder_state)
            str_result += [dataset.idx2str[int(res.item())] for res in output.argmax(-1)[0]]
        print(str_equ, end=' ')
        print(str_result)

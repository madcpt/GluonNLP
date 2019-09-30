import collections
import io
import math

from mxnet import autograd, cpu, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn

from data.Equation import Equation


class Encoder(nn.Block):
    def __init__(self, vocal_size, embed_dim, num_hiddens, 
                num_layers, *args):
        super(Encoder, self).__init__(*args)
        self.embedding = nn.Embedding(vocal_size, embed_dim)
        self.rnn = rnn.GRU(num_hiddens, num_layers)

    def forward(self, inputs, state):
        embedding = self.embedding(inputs.T)
        return self.rnn(embedding, state)
    
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

class Decoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, 
                num_layers, *args):
        super(Decoder, self).__init__(*args)
        self.embedding = nn.Embedding(vocal_size, embed_dim)
        self.rnn = rnn.GRU(num_hiddens, num_layers)
        self.dense = nn.Dense(vocab_size, flatten=False, activation='tanh')
    
    def forward(self, inputs, state):
        # input_and_state = nd.concat(self.embedding(cur_input), encoder_state, dim=1)
        # output, state = self.rnn(input_and_state.expand_dims(0), state)
        x = self.embedding(inputs)
        output, state = self.rnn(x, state)
        y = self.dense(output)
        return y, state

    def begin_state(self, enc_state):
        return enc_state
        
class EncoderDecoder(nn.Block):
    def __init__(self, encoder, decoder, dataloader:Equation, *args):
        super(EncoderDecoder, self).__init__(*args)
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader

    def forward(self, x, y, loss):
        l = 0.0
        n = 0
        # output = model(x, y)
        encoder_state = encoder.begin_state(batch_size=batch_size)
        _, encoder_state = encoder(x, encoder_state)
        decoder_state = decoder.begin_state(encoder_state)
        # y_0, decoder_state = decoder(x.T[-1].expand_dims(0),encoder_state)
        for i in range(y.shape[1]-1):
            y_i = y.T[i]
            output, decoder_state = decoder(y_i.expand_dims(0), decoder_state)
            l = l + loss(output[0], y.T[i+1].reshape(-1,1))  
            n = n + y.shape[0]
            # print(output[0])
            # print(y.T[i+1].reshape(-1,1))
        return l, n
        

if __name__=='__main__':
    ctx = cpu()
    batch_size = 20
    max_len = 10
    dataset = Equation(max_len, ctx)
    vocal_size = len(dataset.idx2str)
    embed_dim = 10
    num_hiddens = 32
    num_epoch = 301

    encoder = Encoder(vocal_size, embed_dim, num_hiddens, 2)
    # encoder.initialize()
    decoder = Decoder(vocal_size, embed_dim, num_hiddens, 2)
    # decoder.initialize()
    model = EncoderDecoder(encoder, decoder, dataset)
    model.initialize()

    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Xavier())
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': 1e0})

    for i in range(num_epoch):
        l = 0.0
        # count = 0
        with autograd.record():
            total_loss = nd.array([0], ctx=ctx)
            for x,y in dataset.generate_data_iter(batch_size):
                l, n = model(x, y, loss)
                # str_result = [dataset.idx2str[int(i.asscalar())] for i in output.argmax(axis=-1)[0]]
                total_loss = total_loss + l.mean()
                # count += n
                # break
        total_loss.backward()
        trainer.step(1)
        if i % 50 == 0:
            print(i, end=', ')
            print(total_loss.asscalar())

    for x,y in dataset.generate_data_iter(1):
        encoder_state = encoder.begin_state(batch_size=1)
        _, encoder_state = encoder(x, encoder_state)
        decoder_state = decoder.begin_state(encoder_state)
        str_equ = [dataset.idx2str[int(i.asscalar())] for i in x[0]]
        str_result = []
        for i in range(y.shape[1]-1):
            y_i = y.T[i]
            output, decoder_state = decoder(y_i.expand_dims(0), decoder_state)
            str_result += [dataset.idx2str[int(i.asscalar())] for i in output.argmax(axis=-1)[0]]
        print(str_equ, end=' ')
        print(str_result)

import math
import time

from mxnet import autograd, gpu, init, nd
from mxnet.gluon import Trainer
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn


def produce_dataset(size):
    X = nd.arange(start=1, stop=size+1, step=1.0)
    return X

data_size = 100
dataset = produce_dataset(data_size)

def data_iter_consecutive(dataset, batch_size, num_steps, ctx=None):
    dataset = nd.array(dataset, ctx=ctx)
    data_len = len(dataset)
    batch_len = data_len // batch_size
    indices = dataset[0 : batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i : i + num_steps]
        Y = indices[:, i + 1 : i + num_steps + 1]
        yield X, Y

# for X, Y in data_iter_consecutive(dataset, 2, 10):
#     print("X: " + str(X))
#     print("Y: " + str(Y))

class RNNModel(nn.Block):
    def __init__(self, rnn_layer, data_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.data_size = data_size
        self.dense = nn.Dense(data_size)
    
    def forward(self, inputs, state, *args):
        X = nd.one_hot(inputs.T, self.data_size)
        Y, state = self.rnn(X, state)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

num_hidden = 32

# rnn_layer = rnn.RNN(num_hidden)
# rnn_layer = rnn.LSTM(num_hidden)
rnn_layer = rnn.GRU(num_hidden)

rnn_layer.initialize()

batch_size = 2
num_steps = 3

ctx = gpu()
model = RNNModel(rnn_layer, data_size)

num_epochs, lr, clipping_theta = 1000, 3e0, 1e-2
pred_period = 50

loss = gloss.SoftmaxCrossEntropyLoss()
model.initialize(force_reinit=True, ctx=ctx, init=init.Normal(0.01))
trainer = Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})

for epoch in range(num_epochs):
    l_sum, n, start = 0.0, 0, time.time()
    data_iter = data_iter_consecutive(dataset, batch_size, num_steps, ctx)
    state = model.begin_state(batch_size=batch_size, ctx=ctx)
    for X, Y in data_iter:
        # print("X: " + str(X))
        # print("Y: " + str(Y))
        for s in state:
            s.detach()
        with autograd.record():
            (output, state) = model(X, state)
            y = Y.T.reshape((-1,))
            l = loss(output, y).mean()
        l.backward()
        params = [p.data() for p in model.collect_params().values()]
        grad_clipping(params, clipping_theta, ctx)
        trainer.step(1)
        l_sum += l.asscalar() * y.size
        n += y.size

    if (epoch + 1) % pred_period == 0:
        print('epoch %d, perplexity %f, time %.2f sec' % (
        epoch + 1, math.exp(l_sum / n), time.time() - start))

# predict
prefixes = [1, 2, 3]
state = model.begin_state(batch_size=1, ctx=ctx)
output = [prefixes[0]]
for i in range(data_size + len(prefixes) - 1):
    X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
    (Y, state) = model(X, state)
    if i < len(prefixes) - 1:
        output.append(prefixes[i + 1])
    else:
        output.append(int(Y.argmax(axis=1).asscalar()))

print(output)

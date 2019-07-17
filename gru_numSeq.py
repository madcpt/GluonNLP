import math
import random
import time
import zipfile

from mxnet import autograd, gluon, gpu, init, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        print(X)
        print(Y)
        yield nd.array(X, ctx), nd.array(Y, ctx)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def grad_clipping(params, theta, ctx):
    """Clip the gradient."""
    if theta is not None:
        norm = nd.array([0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm


class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def produce_dataset(size):
    X = nd.arange(start=1, stop=size+1, step=1.0)
    return X

vocab_size = 100
corpus_indices = produce_dataset(vocab_size)

num_hiddens = 512
batch_size = 10
num_steps = 2

ctx = gpu()

num_epochs, lr, clipping_theta = 500, 1e1, 1e-2
pred_period, pred_len = 30, 100

gru_layer = rnn.GRU(num_hiddens)
model = RNNModel(gru_layer, vocab_size)


loss = gloss.SoftmaxCrossEntropyLoss()
model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': lr, 'momentum': 0, 'wd': 0})

for epoch in range(num_epochs):
    l_sum, n, start = 0.0, 0, time.time()
    data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx)
    # data_iter = data_iter_random(corpus_indices, batch_size, num_steps, ctx)
    state = model.begin_state(batch_size=batch_size, ctx=ctx)
    for X, Y in data_iter:
        for s in state:
            s.detach()
        with autograd.record():
            (output, state) = model(X, state)
            y = Y.T.reshape((-1,))
            l = loss(output, y).mean()
        l.backward()

        # 梯度裁剪
        params = [p.data() for p in model.collect_params().values()]
        grad_clipping(params, clipping_theta, ctx)
        trainer.step(1)  # 因为已经误差取过均值，梯度不用再做平均
        l_sum += l.asscalar() * y.size
        n += y.size

    if (epoch + 1) % pred_period == 0:
        print('epoch %d, perplexity %f, time %.2f sec' % (
            epoch + 1, math.exp(l_sum / n), time.time() - start))
        

# predict
prefixes = [9, 10, 11]
state = model.begin_state(batch_size=1, ctx=ctx)
output = [prefixes[0]]
for i in range(pred_len):
# for i in range(vocab_size + len(prefixes) - 1):
    X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
    (Y, state) = model(X, state)
    if i < len(prefixes) - 1:
        output.append(prefixes[i + 1])
    else:
        output.append(int(Y.argmax(axis=1).asscalar()))

print(output)

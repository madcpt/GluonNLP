import math
import time

from mxnet import autograd, cpu, gluon, gpu, init, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn, rnn


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - num_steps)
    for i in range(epoch_size):
        # i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

def produce_dataset(size):
    X = nd.arange(start=0, stop=size, step=1.0)
    return X

class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocal_size, ctx, *args):
        super(RNNModel, self).__init__(*args)
        self.rnn_layer = rnn_layer
        self.vocal_size = vocal_size
        self.ctx = ctx
        self.dense = nn.Dense(vocal_size)
    
    def forward(self, input, state):
        x = input
        # input is (batch_size, num_step)
        x = nd.one_hot(x.T, depth=self.vocal_size)
        output, state = self.rnn_layer(x, state)
        output = self.dense(output.reshape((-1,output.shape[-1])))
        return output, state
    
    def begin_state(self, *args, **kwargs):
        return self.rnn_layer.begin_state(*args, **kwargs)

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


if __name__=='__main__':
    vocab_size = 100
    corpus_indices = produce_dataset(vocab_size)

    num_hiddens = 128
    batch_size = 3
    num_steps = 3

    try:
        ctx = gpu(0)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = cpu()

    num_epochs, lr, clipping_theta = 200, 1e-1, 1e-2
    pred_period, pred_len = 30, 60

    gru_layer = rnn.GRU(num_hiddens)
    model = RNNModel(gru_layer, vocab_size, ctx=ctx)


    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps)
        for x, Y in data_iter:
            state = model.begin_state(batch_size)
            for s in state:
                s.detach()
            with autograd.record():
                output, state = model(x, state)
                y = Y.T.reshape(-1,)
                l = loss(output, y).mean()
            l.backward()
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size
        params = [p.data() for p in model.collect_params().values()]
        grad_clipping(params, clipping_theta, ctx)
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))

    valid_set = [2,3,4,5,6,7]
    prediction = valid_set.copy()
    for i in range(pred_len-len(valid_set)):
        state = model.begin_state(batch_size=1, ctx=ctx)
        X = nd.array(prediction[-num_steps:], ctx=ctx).reshape((1,num_steps))
        Y, state = model(X, state)
        result = Y.argmax(axis=-1)
        # print(X)
        # print(result)
        # print()
        prediction.append(int(result.reshape(-1,)[-1].asscalar()))

    print(prediction)

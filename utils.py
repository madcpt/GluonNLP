from mxnet import nd
import logging
import time


def get_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None, logger=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = batch_len - num_steps

    if logger is not None:
        logger.debug('batch_size :{}'.format(batch_size))
        logger.debug('batch_len :{}'.format(batch_len))
        logger.debug('epoch_size: {}'.format(epoch_size))
        logger.debug(corpus_indices)
        logger.debug(indices)

    for i in range(epoch_size):
        # i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


if __name__ == '__main__':
    corpus_indices = list(range(11))[1:]
    batch_size = 2
    num_steps = 3

    data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps)
    for i, (X, Y) in enumerate(data_iter):
        logger.debug('epoch {}'.format(i))
        logger.debug('X: {}'.format(str(X)))
        logger.debug('Y: {}'.format(str(Y)))
        logger.debug()
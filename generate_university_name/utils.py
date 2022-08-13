import numpy as np


def load_data():
    data = open('university.txt', 'r').read()
    chars = list(set(data))
    vocab_size = len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (len(data), vocab_size))
    chars = sorted(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    X_text = data.split('\n')
    X_text = [x.strip() for x in X_text]
    return X_text, vocab_size, char_to_ix, ix_to_char


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_initial_loss(n_y, seq_length):
    return -np.log(1.0/n_y) * seq_length


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001
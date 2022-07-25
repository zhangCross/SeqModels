from utils import *
import copy

X_text, vocab_size, char_to_ix, ix_to_char = load_data()
np.random.shuffle(X_text)
n_a = 50  # dimension of rnn state
n_x = vocab_size  # dimension of input
n_y = n_x  # dimension of output
n_samples = len(X_text)
learning_rate = 0.01
iterations = 30000
n_generate = 100
avg_seq_length = np.mean([len(x_text) for x_text in X_text])


def get_onehot(x_text):
    x_oh = {}
    y = []
    for t in range(len(x_text)+1):
        x_oh[t] = np.zeros((n_x, 1))
        if t != 0:
            ch = x_text[t-1]
            ch_ix = char_to_ix[ch]
            x_oh[t][ch_ix] = 1
            y.append(ch_ix)
    y.append(char_to_ix['\n'])
    return x_oh, y


def get_samples_onehot():
    X_oh = []
    Y = []
    for x_text in X_text:
        x_oh, y = get_onehot(x_text)
        X_oh.append(x_oh)
        Y.append(y)
    return X_oh, Y


def initialize_parameters():
    """
    Initialize parameters

    Returns:
    parameters -- python dictionary containing:
                        Wax -- input-to-hidden weights matrix, numpy array of shape (n_a, n_x)
                        Waa -- hidden-to-hidden weights matrix, numpy array of shape (n_a, n_a)
                        Wya -- hidden-to-output weights matrix, numpy array of shape (n_y, n_a)
                        b --  Bias relating to the state, numpy array of shape (n_a, 1)
                        by -- Bias relating to the output, numpy array of shape (n_y, 1)
    """
    Wax = np.random.randn(n_a, n_x) * 0.01
    Waa = np.random.randn(n_a, n_a) * 0.01
    Wya = np.random.randn(n_y, n_a) * 0.01
    b = np.zeros((n_a, 1))
    by = np.zeros((n_y, 1))
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
    return parameters


def rnn_step_forward(parameters, a_prev, xt_oh):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, xt_oh) + np.dot(Waa, a_prev) + b)
    yt_hat = softmax(np.dot(Wya, a_next) + by)
    return a_next, yt_hat


def rnn_forward(x_oh, y, parameters):
    a, y_hat = {}, {}
    a[-1] = np.zeros((n_a, 1))
    loss = 0
    for t in range(len(x_oh)):
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x_oh[t])
        loss -= np.log(y_hat[t][y[t]][0])
    cache = (y_hat, a)
    return loss, cache


def rnn_step_backward(dy, gradients, parameters, xt_oh, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']  # backprop into h
    daraw = (1 - a * a) * da  # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, xt_oh.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients


def rnn_backward(x_oh, y, parameters, cache):
    y_hat, a = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    gradients = {'dWax': np.zeros_like(Wax), 'dWaa': np.zeros_like(Waa), 'dWya': np.zeros_like(Wya),
                 'db': np.zeros_like(b), 'dby': np.zeros_like(by), 'da_next': np.zeros_like(a[0])}
    for t in reversed(range(len(x_oh))):
        dy = np.copy(y_hat[t])
        dy[y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x_oh[t], a[t], a[t - 1])
    return gradients, a


def clip(gradients, threshold):
    #  Clips the gradients' values between -threshold and +threshold.
    gradients = copy.deepcopy(gradients)
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients[
        'dby']
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -threshold, threshold, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients


def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b'] += -lr * gradients['db']
    parameters['by'] += -lr * gradients['dby']
    return parameters


def optimize(x_oh, y, parameters):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    x_oh -- list of one hot vector, each one hot vector maps to a character of x_text.
    y -- list of integers, each integer maps to a character of x_text but shifted one index to the left.

    Returns:
    loss -- cross-entropy loss
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(x)-1] -- the last hidden state, of shape (n_a, 1)
    """
    loss, cache = rnn_forward(x_oh, y, parameters)
    gradients, a = rnn_backward(x_oh, y, parameters, cache)
    gradients = clip(gradients, 5)
    update_parameters(parameters, gradients, learning_rate)
    return loss


def sample_rnn(parameters, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN
    Returns:
    indices -- list of integers, each integer maps to index of one character.
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    xt = np.zeros((n_x, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0
    newline_ix = char_to_ix['\n']

    while idx != newline_ix and counter < 30:
        at = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + b)
        zt = np.dot(Wya, at) + by
        yt = softmax(zt)
        np.random.seed(counter + seed)
        idx = np.random.choice(range(n_y), p=yt.ravel())
        indices.append(idx)
        xt = np.zeros((n_x, 1))
        xt[idx] = 1
        a_prev = at
        counter += 1
    return indices


def get_sample_text(sample_idx):
    sample_text = ''.join(ix_to_char[ix] for ix in sample_idx).strip()
    return sample_text


def model(X_oh, Y, parameters):
    """
    Trains the model.
    """
    loss = get_initial_loss(n_y, avg_seq_length)
    print(loss)
    for i in range(iterations):
        idx = i % n_samples
        x_oh, y = X_oh[idx], Y[idx]
        curr_loss = optimize(x_oh, y, parameters)
        loss = smooth(loss, curr_loss)
        if i % 5000 == 0:
            print('Iteration: %d, Loss: %f' % (i, loss))
    return parameters, loss


def get_n_samples(parameters):
    samples = []
    seed = 0
    counter = 0
    while counter < n_generate:
        sample_idx = sample_rnn(parameters, seed)
        sample_text = get_sample_text(sample_idx)
        seed += 1
        if sample_text in X_text:
            continue
        samples.append(sample_text)
        print(sample_text)
        counter += 1
    return samples


def main():
    X_oh, Y = get_samples_onehot()
    parameters = initialize_parameters()
    parameters, loss = model(X_oh, Y, parameters)
    print("final loss: %f" % loss)
    get_n_samples(parameters)


def get_onehot_test(target):
    x_oh, y = target("IT")
    print(x_oh)
    print(y)


if __name__ == '__main__':
    main()


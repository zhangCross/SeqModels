import tensorflow as tf
from faker import Faker
import random
import numpy as np
from babel.dates import format_date

fake = Faker()
FORMATS = ['short',
           'short',
           'medium',
           'long',
           'long',
           'full',
           'full',

           'yy.M.d',
           'yy.MM.dd',
           'yy/M/d',
           'yy/MM/dd',
           'yyMMdd',

           'yyyy.M.d',
           'yyyy.MM.dd',
           'yyyy/M/d',
           'yyyy/MM/dd',
           'yyyyMMdd',
           
           'yy年M月d日',
           'yy年MM月dd日',
           'yy年MMMMd日',
           'yyyy年M月d日',
           'yyyy年MM月d日',
           'yyyy年MMMMd日',

           'yy年M月d号',
           'yy年MM月dd号',
           'yy年MMMMd号',
           'yyyy年M月d号',
           'yyyy年MM月dd号',
           'yyyy年MMMMd号',

           'yy年M月d日EEE',
           'yy年MM月dd日EEE',
           'yy年MMMMd日EEE',
           'yyyy年M月d日EEE',
           'yyyy年MM月dd日EEE',
           'yyyy年MMMMd日EEE',

           'yy年M月d号EEE',
           'yy年MM月dd号EEE',
           'yy年MMMMd号EEE',
           'yyyy年M月d号EEE',
           'yyyy年MM月dd号EEE',
           'yyyy年MMMMd号EEE',

           'yy年M月d日EEEE',
           'yy年MM月dd日EEEE',
           'yy年MMMMd日EEEE',
           'yyyy年M月d日EEEE',
           'yyyy年MM月dd日EEEE',
           'yyyy年MMMMd日EEEE',

           'yy年M月d号EEEE',
           'yy年MM月dd号EEEE',
           'yy年MMMMd号EEEE',
           'yyyy年M月d号EEEE',
           'yyyy年MM月dd号EEEE',
           'yyyy年MMMMd号EEEE']
FORMATS1 = ['short',
            'medium',
            'long',
            'full',
            'd MMM yyyy',
            'd MMMM yyyy',
            'dd MMM yyyy',
            'd MMM, yyyy',
            'd MMMM, yyyy',
            'dd, MMM yyyy',
            'd MM YY',
            'd MMMM yyyy',
            'MMMM d yyyy',
            'MMMM d, yyyy',
            'dd.MM.YY']


def load_date():
    """
        returns: human-readable string, machine-readable string, and date object
    """
    dt = fake.date_object()
    human_readable = format_date(dt, format=random.choice(FORMATS), locale='zh_CN')  # en_US
    human_readable = human_readable.lower().replace(',', '')
    machine_readable = dt.isoformat()
    return human_readable, machine_readable, dt


def load_dataset(m):
    """
        m: the number of examples to generate
    """

    human_vocab = set()
    machine_vocab = set()
    dataset = []
    for i in range(m):
        h, m, _ = load_date()
        dataset.append((h, m))
        human_vocab.update(set(h))
        machine_vocab.update(set(m))
    inv_human = dict(enumerate(sorted(human_vocab) + ['<unk>', '<pad>']))
    human = {v: k for k, v in inv_human.items()}
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v: k for k, v in inv_machine.items()}
    return dataset, human, machine, inv_machine


def string_to_int(string, length, vocab):
    """
    Converts a string to char indices
    """
    string = string.lower().replace(',', '')
    if len(string) > length:
        string = string[:length]
    unk_idx = vocab.get('<unk>')
    rep = list(map(lambda x: vocab.get(x, unk_idx), string))
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    return rep


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset)
    X = [string_to_int(x, Tx, human_vocab) for x in X]
    Y = [string_to_int(y, Ty, machine_vocab) for y in Y]
    hv_cnt = len(human_vocab)
    mv_cnt = len(machine_vocab)
    Xoh = np.array(list(map(lambda x: tf.keras.utils.to_categorical(x, num_classes=hv_cnt), X)))
    Yoh = np.array(list(map(lambda x: tf.keras.utils.to_categorical(x, num_classes=mv_cnt), Y)))
    return Xoh, Yoh


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = tf.keras.backend.ndim(x)
    if ndim == 2:
        return tf.keras.backend.softmax(x)
    elif ndim > 2:
        e = tf.keras.backend.exp(x - tf.keras.backend.max(x, axis=axis, keepdims=True))
        s = tf.keras.backend.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def main():
    s = string_to_int("123", 5, {"1": 1, "2": 2, "<pad>": 0, "<unk>": 5})
    print(s)


if __name__ == '__main__':
    main()

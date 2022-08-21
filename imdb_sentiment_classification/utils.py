import time
import tensorflow as tf
import matplotlib.pyplot as plt
imdb = tf.keras.datasets.imdb
vocab_size = 10000


def get_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
    print("train shape: %s, test shape: %s" % (train_data.shape, test_data.shape))
    return (train_data, train_labels), (test_data, test_labels)


def get_index_word():
    word_index = imdb.get_word_index()
    word_index = {k: (v+3) for k, v in word_index.items() if v+3 < vocab_size}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    index_word = {v: k for k, v in word_index.items()}
    return index_word, word_index


def train_model(model, train_data, train_labels, epochs=20, batch_size=512):
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    start_time = time.time()
    history = model.fit(partial_x_train, partial_y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    print("train time %s seconds" % (time.time() - start_time))
    return history


def test_model(model, test_data, test_labels):
    results = model.evaluate(test_data, test_labels, verbose=0)
    print(results)
#[0.3066553473472595, 0.8744000792503357]
#[0.33859536051750183, 0.8532400727272034] 25, 64


def plot_history(history):
    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'b--', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'b--', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
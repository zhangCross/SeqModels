import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
imdb = tf.keras.datasets.imdb


def get_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
    print("train shape:", train_data.shape)
    return (train_data, train_labels), (test_data, test_labels)


def get_index_word():
    word_index = imdb.get_word_index()
    word_index = {k: (v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    index_word = {v: k for k, v in word_index.items()}
    return index_word, word_index


def decode_text(indices):
    index_word, _ = get_index_word()
    return " ".join([index_word.get(i) for i in indices])


def init_model():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 32))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels):
    train_data = pad_sequences(train_data, value=0, padding='post', maxlen=256)
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    start_time = time.time()
    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val),
                        verbose=0)
    print("train time %s seconds" % (time.time() - start_time))
    return history


def test_mode(model, test_data, test_labels):
    test_data = pad_sequences(test_data, value=0, padding='post', maxlen=256)
    results = model.evaluate(test_data, test_labels, verbose=0)
    print(results)


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


def main():
    (train_data, train_labels), (test_data, test_labels) = get_data()
    model = init_model()
    history = train_model(model, train_data, train_labels)
    plot_history(history)
    test_mode(model, test_data, test_labels)


if __name__ == '__main__':
    main()




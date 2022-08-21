import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
vocab_size = 10000
imdb = tf.keras.datasets.imdb


def get_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
    print("train shape: %s, test shape: %s" % (train_data.shape, test_data.shape))
    return (train_data, train_labels), (test_data, test_labels)


def get_index_word():
    # only for decode
    word_index = imdb.get_word_index()
    word_index = {k: (v+3) for k, v in word_index.items() if v+3 < vocab_size}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    index_word = {v: k for k, v in word_index.items()}
    return index_word, word_index


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
    return word_to_vec_map


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].strip().lower().split()
        for j, w in enumerate(sentence_words):
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- words to their GloVe vectors.
    word_to_index -- words to their indices

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]
    emb_matrix = np.zeros((vocab_size, emb_dim))
    for word, idx in word_to_index.items():
        word = word.strip("'")
        if word in word_to_vec_map:
            emb_matrix[idx, :] = word_to_vec_map[word]
        elif word == "hadn't":
            emb_matrix[idx, :] = word_to_vec_map["not"]
        elif word == "t'aime":
            emb_matrix[idx, :] = word_to_vec_map["love"]
        elif "'" in word:
            ori_word = word.split("'")[0]
            if ori_word in word_to_vec_map:
                emb_matrix[idx, :] = word_to_vec_map[ori_word]
        elif "´" in word:
            ori_word = word.split("´")[0]
            if ori_word in word_to_vec_map:
                emb_matrix[idx, :] = word_to_vec_map[ori_word]
    embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer


def get_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the model.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- every word to its 50-dimensional vector
    word_to_index -- words to their indices

    Returns:
    model -- a model instance in Keras
    """
    sentence_indices = tf.keras.layers.Input(input_shape, dtype='int32')
    print(sentence_indices.shape)
    # embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embedding_layer = tf.keras.layers.Embedding(vocab_size, 32)
    X = embedding_layer(sentence_indices)
    X = tf.keras.layers.LSTM(128, return_sequences=True)(X)
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    X = tf.keras.layers.LSTM(128)(X)
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.models.Model(inputs=sentence_indices, outputs=X)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels):
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    start_time = time.time()
    history = model.fit(partial_x_train, partial_y_train, epochs=25, batch_size=32, validation_data=(x_val, y_val))
    print("train time %s seconds" % (time.time() - start_time))
    return history


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


def test_mode(model, test_data, test_labels):
    results = model.evaluate(test_data, test_labels, verbose=0)
    print(results)


def main():
    MAXLEN = 256
    (train_data, train_labels), (test_data, test_labels) = get_data()
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post', maxlen=MAXLEN)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, value=0, padding='post', maxlen=MAXLEN)
    word_to_vec_map = read_glove_vecs('glove.50d.txt')
    index_word, word_index = get_index_word()
    model = get_model((MAXLEN,), word_to_vec_map, word_index)
    model.summary()
    history = train_model(model, train_data, train_labels)
    plot_history(history)
    # test_mode(model, test_data, test_labels)


if __name__ == '__main__':
    main()


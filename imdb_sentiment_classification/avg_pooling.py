import tensorflow as tf
import utils


def decode_text(indices):
    index_word, _ = utils.get_index_word()
    return " ".join([index_word.get(i) for i in indices])


def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(utils.vocab_size, 16))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    (train_data, train_labels), (test_data, test_labels) = utils.get_data()
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post', maxlen=256)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, value=0, padding='post', maxlen=256)
    # print(decode_text(train_data[0]))
    model = get_model()
    model.summary()
    history = utils.train_model(model, train_data, train_labels)
    utils.plot_history(history)
    utils.test_model(model, test_data, test_labels)


if __name__ == '__main__':
    main()




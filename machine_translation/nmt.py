from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

Tx = 30
Ty = 10
repeator = tf.keras.layers.RepeatVector(Tx)
concatenator = tf.keras.layers.Concatenate(axis=-1)
densor1 = tf.keras.layers.Dense(10, activation="tanh")
densor2 = tf.keras.layers.Dense(1, activation="relu")
activator = tf.keras.layers.Activation(softmax, name='attention_weights')
dotor = tf.keras.layers.Dot(axes=1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector.
    Arguments:
    a -- hidden state output of the Bi-LSTM, shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, shape (m, n_s)
    Returns:
    context -- context vector, input of the succeeding decoder
    """
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])  # (m, Tx, 2*n_a + n_s)
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context


def modelf(Tx, Ty, n_a, n_s, src_vocab_size, target_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    src_vocab_size -- size of human vocab
    target_vocab_size -- size of machine vocab

    Returns:
    model -- model instance
    """
    X = tf.keras.layers.Input(shape=(Tx, src_vocab_size))
    outputs = []
    a = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_a, return_sequences=True))(X)
    s0 = tf.keras.layers.Input(shape=(n_s,), name='s0')
    c0 = tf.keras.layers.Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    post_lstm_cell = tf.keras.layers.LSTM(n_s, return_state=True)
    output_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_lstm_cell(inputs=context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)
    model = tf.keras.models.Model(inputs=[X, s0, c0], outputs=outputs)
    return model


def predict(model, n_s, human_vocab, inv_machine_vocab):
    EXAMPLES = ['1999年12月1日', '98年1月2日', '2022/11/11', '2021年10月30日', '2000年三月7日',
                 '2009.11.09', '2022年8月27日星期六', '111111']
    s00 = np.zeros((1, n_s))
    c00 = np.zeros((1, n_s))
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(
            list(map(lambda x: tf.keras.utils.to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
        source = np.swapaxes(source, 0, 1)
        source = np.expand_dims(source, axis=0)
        prediction = model.predict([source, s00, c00])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]
        print("source:", example)
        print("output:", ''.join(output), "\n")


def plot_attention_map(model, src_vocab, inv_target_vocab, text, n_s=128, num=7):
    """
    Plot the attention map.
    """
    attention_map = np.zeros((Ty, Tx))
    X = model.inputs[0]
    s0 = model.inputs[1]
    c0 = model.inputs[2]
    s = s0
    c = s0
    a = model.layers[2](X)
    outputs = []
    for t in range(Ty):
        s_prev = s
        s_prev = model.layers[3](s_prev)
        concat = model.layers[4]([a, s_prev])
        e = model.layers[5](concat)
        energies = model.layers[6](e)
        alphas = model.layers[7](energies)
        context = model.layers[8]([alphas, a])
        s, _, c = model.layers[10](context, initial_state=[s, c])
        outputs.append(energies)

    f = tf.keras.models.Model(inputs=[X, s0, c0], outputs=outputs)
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    X = np.array(string_to_int(text, Tx, src_vocab)).reshape((1, 30))
    Xoh = np.array(list(map(lambda x: tf.keras.utils.to_categorical(x, num_classes=len(src_vocab)), X)))
    r = f([Xoh, s0, c0])

    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = r[t][0, t_prime]
    row_max = attention_map.max(axis=1)[:, None]
    attention_map = np.divide(attention_map, row_max, out=np.zeros_like(attention_map), where=row_max != 0)
    prediction = model.predict([Xoh, s0, c0])
    predicted_text = []
    for i in range(len(prediction)):
        predicted_text.append(int(np.argmax(prediction[i], axis=1)))
    predicted_text = list(predicted_text)
    predicted_text = [inv_target_vocab[i] for i in predicted_text]
    input_tokens = [ch for ch in text]
    input_length = len(text)
    output_length = Ty

    f, ax = plt.subplots(figsize=(8, 8))
    # add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')
    # add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text)
    ax.set_xticks(range(input_length))
    ax.set_xticklabels(input_tokens, rotation=45)
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')
    plt.grid()
    plt.show()
    return attention_map


def main():
    m = 100000 # 160000
    n_a = 32  # number of units for the pre-attention LSTM
    n_s = 64  # number of units for the post-attention LSTM
    dataset, src_vocab, target_vocab, inv_target_vocab = load_dataset(m)
    # print(dataset[:10])
    Xoh, Yoh = preprocess_data(dataset, src_vocab, target_vocab, Tx, Ty)
    model = modelf(Tx, Ty, n_a, n_s, len(src_vocab), len(target_vocab))
    # model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))
    # model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=100)
    # model.save_weights("model.h5")
    model.load_weights('model.h5')
    predict(model, n_s, src_vocab, inv_target_vocab)
    attention_map = plot_attention_map(model, src_vocab, inv_target_vocab, "2019年8月31日周六", num=7, n_s=64)


if __name__ == '__main__':
    main()

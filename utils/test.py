import tensorflow as tf


class MyModel(tf.keras.models.Model):
    def __int__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.embedding = tf.keras.layers.Embedding(100, 16)
        self.g_avg_pool = tf.keras.layers.GlobalAvgPool1D()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.do = tf.keras.layers.Dropout(0.2)
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.d1(x)
        x = self.do(x)
        x = self.d2(x)
        return x


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer=tf.random_normal_initializer,
                                 trainable=True)
        self.b = self.add_weight(shape=(units,),
                                 initializer=tf.random_normal_initializer,
                                 trainable=True)

    def call(self, inputs, training=None, mask=None):
        return tf.matmul(inputs, self.w) + self.b


def get_model():
    inputs = tf.keras.Input(shape=(256,))
    emb = tf.keras.layers.Embedding(1000, 16)(inputs)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(emb)
    d1 = tf.keras.layers.Dense(16, activation="relu")(avg_pool)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(d1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)


def get_model_seq():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(100, 16))
    model.add(tf.keras.layers.GlobalAvgPool1D())
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.summary()
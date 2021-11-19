import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras import Input, Model
from keras.callbacks import ModelCheckpoint



class TweetClassifier:

    def __init__(self, tokenizer, bert_layer, max_len, lr=0.0001,
                 epochs=15, batch_size=32,
                 activation='sigmoid', optimizer='SGD',
                 beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                 metrics='accuracy', loss='binary_crossentropy'):

        self.lr = lr
        self.epochs = epochs
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.bert_layer = bert_layer

        self.activation = activation
        self.optimizer = optimizer

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.metrics = metrics
        self.loss = loss

    def encode(self, texts):

        all_tokens = []
        masks = []
        segments = []

        for text in texts:
            tokenized = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]'])

            len_zeros = self.max_len - len(tokenized)

            padded = tokenized + [0] * len_zeros
            mask = [1] * len(tokenized) + [0] * len_zeros
            segment = [0] * self.max_len

            all_tokens.append(padded)
            masks.append(mask)
            segments.append(segment)

        return np.array(all_tokens), np.array(masks), np.array(segments)

    def make_model(self):

        # Shaping the inputs to our model

        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name='input_ids')

        input_mask = Input(shape=(self.max_len,), dtype=tf.int32, name='input_mask')

        segment_ids = Input(shape=(self.max_len,), dtype=tf.int32, name='segment_ids')

        pooled_output, sequence_output = self.bert_layer([input_ids, input_mask, segment_ids])

        clf_output = sequence_output[:, 0, :]

        out = tf.keras.layers.Dense(1, activation=self.activation)(clf_output)

        model = Model(inputs=[input_ids, input_mask, segment_ids], outputs=out)

        # define the optimizer

        if self.optimizer is 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

        elif self.optimizer is 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metrics])

        print('Model is compiled with {} optimizer'.format(self.optimizer))

        return model

    def train(self, x):

        checkpoint = ModelCheckpoint('model.h5', monitor='val_loss',
                                     save_best_only=True)

        model = self.make_model()

        X = self.encode(x['cleaned_text'])
        Y = x['target']

        model.fit(X, Y, shuffle=True, validation_split=0.2,
                  batch_size=self.batch_size, epochs=self.epochs,
                  callbacks=[checkpoint])

        print('Model is fit!')

    def predict(self, x):

        X_test_encoded = self.encode(x['cleaned_text'])
        best_model = tf.keras.models.load_model('model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        y_pred = best_model.predict(X_test_encoded)

        return y_pred

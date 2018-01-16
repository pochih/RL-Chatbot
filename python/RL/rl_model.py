# coding=utf-8

import tensorflow as tf
import numpy as np

class PolicyGradient_chatbot():
    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size, n_encode_lstm_step, n_decode_lstm_step, bias_init_vector=None, lr=0.0001):
        self.dim_wordvec = dim_wordvec
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_words = n_words
        self.n_encode_lstm_step = n_encode_lstm_step
        self.n_decode_lstm_step = n_decode_lstm_step
        self.lr = lr

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_vector_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='encode_vector_W')
        self.encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_decode_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step+1])

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b ) # (batch_size*n_encode_lstm_step, dim_hidden)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        reward = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        entropies = []
        loss = 0.
        pg_loss = 0.  # policy gradient loss

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_encode_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)
                # states.append(state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_decode_lstm_step):
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:, i]
            entropies.append(cross_entropy)
            pg_cross_entropy = cross_entropy * reward[:, i]

            pg_current_loss = tf.reduce_sum(pg_cross_entropy) / self.batch_size
            pg_loss = pg_loss + pg_current_loss

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(pg_loss)

        input_tensors = {
            'word_vectors': word_vectors,
            'caption': caption,
            'caption_mask': caption_mask,
            'reward': reward
        }

        feats = {
            'entropies': entropies
        }

        return train_op, pg_loss, input_tensors, feats

    def build_generator(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []
        states = []

        for i in range(0, self.n_encode_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)
                states.append(state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_decode_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:
                # <bos>
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

            embeds.append(current_embed)

        feats = {
            'probs': probs,
            'embeds': embeds,
            'states': states
        }

        return word_vectors, generated_words, feats

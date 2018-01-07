#-*- coding: utf-8 -*-

from __future__ import print_function

import os
import time
import sys
import copy

sys.path.append("python")
from model import Seq2Seq_chatbot
from data_reader import Data_Reader
import data_parser
import config
import re

from gensim.models import KeyedVectors
from rl_model import PolicyGradient_chatbot
from scipy import spatial
import tensorflow as tf
import numpy as np
import math


### Global Parameters ###
checkpoint = config.CHECKPOINT
model_path = config.train_model_path
model_name = config.train_model_name
start_epoch = config.start_epoch
start_batch = config.start_batch

# reversed model
reversed_model_path = config.reversed_model_path
reversed_model_name = config.reversed_model_name

word_count_threshold = config.WC_threshold
r_word_count_threshold = config.reversed_WC_threshold

# dialog simulation turns
max_turns = config.MAX_TURNS

dull_set = ["I don't know what you're talking about.", "I don't know.", "You don't know.", "You know what I mean.", "I know what you mean.", "You know what I'm saying.", "You don't know anything."]

### Train Parameters ###
training_type = config.training_type    # 'normal' for seq2seq training, 'pg' for policy gradient

dim_wordvec = 300
dim_hidden = 1000

n_encode_lstm_step = 22 + 22
n_decode_lstm_step = 22

r_n_encode_lstm_step = 22
r_n_decode_lstm_step = 22

learning_rate = 0.0001
epochs = 500
batch_size = config.batch_size
reversed_batch_size = config.batch_size

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

def make_batch_X(batch_X, n_encode_lstm_step, dim_wordvec, word_vector, noise=False):
    for i in range(len(batch_X)):
        batch_X[i] = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in batch_X[i]]
        if noise:
            batch_X[i].insert(0, np.random.normal(size=(dim_wordvec,))) # insert random normal at the first step

        if len(batch_X[i]) > n_encode_lstm_step:
            batch_X[i] = batch_X[i][:n_encode_lstm_step]
        else:
            for _ in range(len(batch_X[i]), n_encode_lstm_step):
                batch_X[i].append(np.zeros(dim_wordvec))

    current_feats = np.array(batch_X)
    return current_feats

def make_batch_Y(batch_Y, wordtoix, n_decode_lstm_step):
    current_captions = batch_Y
    current_captions = map(lambda x: '<bos> ' + x, current_captions)
    current_captions = map(lambda x: x.replace('.', ''), current_captions)
    current_captions = map(lambda x: x.replace(',', ''), current_captions)
    current_captions = map(lambda x: x.replace('"', ''), current_captions)
    current_captions = map(lambda x: x.replace('\n', ''), current_captions)
    current_captions = map(lambda x: x.replace('?', ''), current_captions)
    current_captions = map(lambda x: x.replace('!', ''), current_captions)
    current_captions = map(lambda x: x.replace('\\', ''), current_captions)
    current_captions = map(lambda x: x.replace('/', ''), current_captions)

    for idx, each_cap in enumerate(current_captions):
        word = each_cap.lower().split(' ')
        if len(word) < n_decode_lstm_step:
            current_captions[idx] = current_captions[idx] + ' <eos>'
        else:
            new_word = ''
            for i in range(n_decode_lstm_step-1):
                new_word = new_word + word[i] + ' '
            current_captions[idx] = new_word + '<eos>'

    current_caption_ind = []
    for cap in current_captions:
        current_word_ind = []
        for word in cap.lower().split(' '):
            if word in wordtoix:
                current_word_ind.append(wordtoix[word])
            else:
                current_word_ind.append(wordtoix['<unk>'])
        current_caption_ind.append(current_word_ind)

    current_caption_matrix = pad_sequences(current_caption_ind, padding='post', maxlen=n_decode_lstm_step)
    current_caption_matrix = np.hstack([current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array(map(lambda x: (x != 0).sum() + 1, current_caption_matrix))

    for ind, row in enumerate(current_caption_masks):
        row[:nonzeros[ind]] = 1

    return current_caption_matrix, current_caption_masks

def index2sentence(generated_word_index, prob_logit, ixtoword):
    # remove <unk> to second high prob. word
    for i in range(len(generated_word_index)):
        if generated_word_index[i] == 3 or generated_word_index[i] <= 1:
            sort_prob_logit = sorted(prob_logit[i])
            curindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
            count = 1
            while curindex <= 3:
                curindex = np.where(prob_logit[i] == sort_prob_logit[(-2)-count])[0][0]
                count += 1

            generated_word_index[i] = curindex

    generated_words = []
    for ind in generated_word_index:
        generated_words.append(ixtoword[ind])

    # generate sentence
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)

    # modify the output sentence 
    generated_sentence = generated_sentence.replace('<bos> ', '')
    generated_sentence = generated_sentence.replace('<eos>', '')
    generated_sentence = generated_sentence.replace(' <eos>', '')
    generated_sentence = generated_sentence.replace('--', '')
    generated_sentence = generated_sentence.split('  ')
    for i in range(len(generated_sentence)):
        generated_sentence[i] = generated_sentence[i].strip()
        if len(generated_sentence[i]) > 1:
            generated_sentence[i] = generated_sentence[i][0].upper() + generated_sentence[i][1:] + '.'
        else:
            generated_sentence[i] = generated_sentence[i].upper()
    generated_sentence = ' '.join(generated_sentence)
    generated_sentence = generated_sentence.replace(' i ', ' I ')
    generated_sentence = generated_sentence.replace("i'm", "I'm")
    generated_sentence = generated_sentence.replace("i'd", "I'd")

    return generated_sentence

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def count_rewards(dull_loss, forward_entropy, backward_entropy, forward_target, backward_target, reward_type='pg'):
    ''' args:
            generated_word_indexs:  <type 'numpy.ndarray'>  
                                    word indexs generated by pre-trained model
                                    shape: (batch_size, n_decode_lstm_step)
            inference_feats:        <type 'dict'>  
                                    some features generated during inference
                                    keys:
                                        'probs': 
                                            shape: (n_decode_lstm_step, batch_size, n_words)
                                        'embeds': 
                                            shape: (n_decode_lstm_step, batch_size, dim_hidden)
                                            current word embeddings at each decode stage
                                        'states': 
                                            shape: (n_encode_lstm_step, batch_size, dim_hidden)
                                            LSTM_1's hidden state at each encode stage
    '''

    # normal training, rewards all equal to 1
    if reward_type == 'normal':
        return np.ones([batch_size, n_decode_lstm_step])

    if reward_type == 'pg':
        forward_entropy = np.array(forward_entropy).reshape(batch_size, n_decode_lstm_step)
        backward_entropy = np.array(backward_entropy).reshape(batch_size, n_decode_lstm_step)
        total_loss = np.zeros([batch_size, n_decode_lstm_step])

        for i in range(batch_size):
            # ease of answering
            total_loss[i, :] += dull_loss[i]
    
            # information flow
            # cosine_sim = 1 - spatial.distance.cosine(embeds[0][-1], embeds[1][-1])
            # IF = cosine_sim * (-1)
    
            # semantic coherence
            forward_len = len(forward_target[i].split())
            backward_len = len(backward_target[i].split())
            if forward_len > 0:
                total_loss[i, :] += (np.sum(forward_entropy[i]) / forward_len)
            if backward_len > 0:
                total_loss[i, :] += (np.sum(backward_entropy[i]) / backward_len)

        total_loss = sigmoid(total_loss) * 1.1

        return total_loss

def train():
    global dull_set

    wordtoix, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)

    if len(dull_set) > batch_size:
        dull_set = dull_set[:batch_size]
    else:
        for _ in range(len(dull_set), batch_size):
            dull_set.append('')
    dull_matrix, dull_mask = make_batch_Y(
                                batch_Y=dull_set, 
                                wordtoix=wordtoix, 
                                n_decode_lstm_step=n_decode_lstm_step)

    ones_reward = np.ones([batch_size, n_decode_lstm_step])

    g1 = tf.Graph()
    g2 = tf.Graph()

    default_graph = tf.get_default_graph() 

    with g1.as_default():
        model = PolicyGradient_chatbot(
                dim_wordvec=dim_wordvec,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_encode_lstm_step=n_encode_lstm_step,
                n_decode_lstm_step=n_decode_lstm_step,
                bias_init_vector=bias_init_vector,
                lr=learning_rate)
        train_op, loss, input_tensors, inter_value = model.build_model()
        tf_states, tf_actions, tf_feats = model.build_generator()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver(max_to_keep=100)
        if checkpoint:
            print("Use Model {}.".format(model_name))
            saver.restore(sess, os.path.join(model_path, model_name))
            print("Model {} restored.".format(model_name))
        else:
            print("Restart training...")
            tf.global_variables_initializer().run()

    r_wordtoix, r_ixtoword, r_bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=r_word_count_threshold)
    with g2.as_default():
        reversed_model = Seq2Seq_chatbot(
            dim_wordvec=dim_wordvec,
            n_words=len(r_wordtoix),
            dim_hidden=dim_hidden,
            batch_size=reversed_batch_size,
            n_encode_lstm_step=r_n_encode_lstm_step,
            n_decode_lstm_step=r_n_decode_lstm_step,
            bias_init_vector=r_bias_init_vector,
            lr=learning_rate)
        _, _, word_vectors, caption, caption_mask, reverse_inter = reversed_model.build_model()
        sess2 = tf.InteractiveSession()
        saver2 = tf.train.Saver()
        saver2.restore(sess2, os.path.join(reversed_model_path, reversed_model_name))
        print("Reversed model {} restored.".format(reversed_model_name))


    dr = Data_Reader(cur_train_index=config.cur_train_index, load_list=config.load_list)

    for epoch in range(start_epoch, epochs):
        n_batch = dr.get_batch_num(batch_size)
        sb = start_batch if epoch == start_epoch else 0
        for batch in range(sb, n_batch):
            start_time = time.time()

            batch_X, batch_Y, former = dr.generate_training_batch_with_former(batch_size)

            current_feats = make_batch_X(
                            batch_X=copy.deepcopy(batch_X), 
                            n_encode_lstm_step=n_encode_lstm_step, 
                            dim_wordvec=dim_wordvec,
                            word_vector=word_vector)

            current_caption_matrix, current_caption_masks = make_batch_Y(
                                                                batch_Y=copy.deepcopy(batch_Y), 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)

            if training_type == 'pg':
                # action: generate batch_size sents
                action_word_indexs, inference_feats = sess.run([tf_actions, tf_feats],
                                                                feed_dict={
                                                                   tf_states: current_feats
                                                                })
                action_word_indexs = np.array(action_word_indexs).reshape(batch_size, n_decode_lstm_step)
                action_probs = np.array(inference_feats['probs']).reshape(batch_size, n_decode_lstm_step, -1)

                actions = []
                actions_list = []
                for i in range(len(action_word_indexs)):
                    action = index2sentence(
                                generated_word_index=action_word_indexs[i], 
                                prob_logit=action_probs[i],
                                ixtoword=ixtoword)
                    actions.append(action)
                    actions_list.append(action.split())

                action_feats = make_batch_X(
                                batch_X=copy.deepcopy(actions_list), 
                                n_encode_lstm_step=n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)

                action_caption_matrix, action_caption_masks = make_batch_Y(
                                                                batch_Y=copy.deepcopy(actions), 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)

                # ease of answering
                dull_loss = []
                for vector in action_feats:
                    action_batch_X = np.array([vector for _ in range(batch_size)])
                    d_loss = sess.run(loss,
                                 feed_dict={
                                    input_tensors['word_vectors']: action_batch_X,
                                    input_tensors['caption']: dull_matrix,
                                    input_tensors['caption_mask']: dull_mask,
                                    input_tensors['reward']: ones_reward
                                })
                    d_loss = d_loss * -1. / len(dull_set)
                    dull_loss.append(d_loss)

                # Information Flow
                pass

                # semantic coherence
                forward_inter = sess.run(inter_value,
                                 feed_dict={
                                    input_tensors['word_vectors']: current_feats,
                                    input_tensors['caption']: action_caption_matrix,
                                    input_tensors['caption_mask']: action_caption_masks,
                                    input_tensors['reward']: ones_reward
                                })
                forward_entropies = forward_inter['entropies']
                former_caption_matrix, former_caption_masks = make_batch_Y(
                                                                batch_Y=copy.deepcopy(former), 
                                                                wordtoix=wordtoix, 
                                                                n_decode_lstm_step=n_decode_lstm_step)
                action_feats = make_batch_X(
                                batch_X=copy.deepcopy(actions_list), 
                                n_encode_lstm_step=r_n_encode_lstm_step, 
                                dim_wordvec=dim_wordvec,
                                word_vector=word_vector)
                backward_inter = sess2.run(reverse_inter,
                                 feed_dict={
                                    word_vectors: action_feats,
                                    caption: former_caption_matrix,
                                    caption_mask: former_caption_masks
                                })
                backward_entropies = backward_inter['entropies']

                # reward: count goodness of actions
                rewards = count_rewards(dull_loss, forward_entropies, backward_entropies, actions, former, reward_type='pg')
    
                # policy gradient: train batch with rewards
                if batch % 10 == 0:
                    _, loss_val = sess.run(
                            [train_op, loss],
                            feed_dict={
                                input_tensors['word_vectors']: current_feats,
                                input_tensors['caption']: current_caption_matrix,
                                input_tensors['caption_mask']: current_caption_masks,
                                input_tensors['reward']: rewards
                            })
                    print("Epoch: {}, batch: {}, loss: {}, Elapsed time: {}".format(epoch, batch, loss_val, time.time() - start_time))
                else:
                    _ = sess.run(train_op,
                                 feed_dict={
                                    input_tensors['word_vectors']: current_feats,
                                    input_tensors['caption']: current_caption_matrix,
                                    input_tensors['caption_mask']: current_caption_masks,
                                    input_tensors['reward']: rewards
                                })
                if batch % 1000 == 0 and batch != 0:
                    print("Epoch {} batch {} is done. Saving the model ...".format(epoch, batch))
                    saver.save(sess, os.path.join(model_path, 'model-{}-{}'.format(epoch, batch)))
            if training_type == 'normal':
                if batch % 10 == 0:
                    _, loss_val = sess.run(
                            [train_op, loss],
                            feed_dict={
                                input_tensors['word_vectors']: current_feats,
                                input_tensors['caption']: current_caption_matrix,
                                input_tensors['caption_mask']: current_caption_masks,
                                input_tensors['reward']: ones_reward
                            })
                    print("Epoch: {}, batch: {}, loss: {}, Elapsed time: {}".format(epoch, batch, loss_val, time.time() - start_time))
                else:
                    _ = sess.run(train_op,
                                 feed_dict={
                                    input_tensors['word_vectors']: current_feats,
                                    input_tensors['caption']: current_caption_matrix,
                                    input_tensors['caption_mask']: current_caption_masks,
                                    input_tensors['reward']: ones_reward
                                })

        print("Epoch ", epoch, " is done. Saving the model ...")
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

if __name__ == "__main__":
    train()

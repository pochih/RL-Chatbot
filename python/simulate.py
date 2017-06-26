#-*- coding: utf-8 -*-

from __future__ import print_function

from gensim.models import KeyedVectors
import data_parser
import config

from model import Seq2Seq_chatbot
import tensorflow as tf
import numpy as np

import re
import os
import sys
import time


#=====================================================
# Global Parameters
#=====================================================
default_model_path = './model/model-20'
default_simulate_type = 1  # type 1 use one former sent, type 2 use two former sents

testing_data_path = 'sample_input.txt' if len(sys.argv) <= 3 else sys.argv[3]
output_path = 'sample_dialog_output.txt' if len(sys.argv) <= 4 else sys.argv[4]

max_turns = config.MAX_TURNS
word_count_threshold = config.WC_threshold

#=====================================================
# Train Parameters
#=====================================================
dim_wordvec = 300
dim_hidden = 1000

n_encode_lstm_step = 22  # need to plus 1 later, because one random normal as the first timestep
n_decode_lstm_step = 22

batch_size = 1

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

def generate_question_vector(state, word_vector, dim_wordvec, n_encode_lstm_step):
    state = [refine(w) for w in state.lower().split()]
    state = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in state]
    state.insert(0, np.random.normal(size=(dim_wordvec,))) # insert random normal at the first step

    if len(state) > n_encode_lstm_step:
        state = state[:n_encode_lstm_step]
    else:
        for _ in range(len(state), n_encode_lstm_step):
            state.append(np.zeros(dim_wordvec))

    return np.array([state]) # 1 x n_encode_lstm_step x dim_wordvec

def generate_answer_sentence(generated_word_index, prob_logit, ixtoword):
    # remove <unk> to second high prob. word
    for i in range(len(generated_word_index)):
        if generated_word_index[i] == 3:
            sort_prob_logit = sorted(prob_logit[i][0])
            # print('max val', sort_prob_logit[-1])
            # print('second max val', sort_prob_logit[-2])
            maxindex = np.where(prob_logit[i][0] == sort_prob_logit[-1])[0][0]
            secmaxindex = np.where(prob_logit[i][0] == sort_prob_logit[-2])[0][0]
            # print('max ind', maxindex, ixtoword[maxindex])
            # print('second max ind', secmaxindex, ixtoword[secmaxindex])
            generated_word_index[i] = secmaxindex

    generated_words = []
    for ind in generated_word_index:
        generated_words.append(ixtoword[ind])

    # generate sentence
    punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)

    # modify the output sentence 
    generated_sentence = generated_sentence.replace('<bos> ', '')
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

def init_history(simulate_type, start_sentence):
    history = []
    history += ['' for _ in range(simulate_type-1)]
    history.append(start_sentence)
    return history

def get_cur_state(simulate_type, dialog_history):
    return ' '.join(dialog_history[-1*simulate_type:]).strip()

def simulate(model_path=default_model_path, simulate_type=default_simulate_type):
    ''' args:
            model_path:     <type 'str'> the pre-trained model using for inference
            simulate_type:  <type 'int'> how many former sents should use as state
    '''

    testing_data = open(testing_data_path, 'r').read().split('\n')

    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)

    _, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)

    model = Seq2Seq_chatbot(
            dim_wordvec=dim_wordvec,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_encode_lstm_step=n_encode_lstm_step,
            n_decode_lstm_step=n_decode_lstm_step,
            bias_init_vector=bias_init_vector)

    word_vectors, caption_tf, probs, _ = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    try:
        print('\n=== Use model {} ===\n'.format(model_path))
        saver.restore(sess, model_path)
    except:
        print('\nUse default model\n')
        saver.restore(sess, default_model_path)

    with open(output_path, 'w') as out:
        for idx, start_sentence in enumerate(testing_data):
            print('dialog {}'.format(idx))
            print('A => {}'.format(start_sentence))
            out.write('dialog {}\nA: {}\n'.format(idx, start_sentence))

            dialog_history = init_history(simulate_type, start_sentence)

            for turn in range(max_turns):
                question = generate_question_vector(state=get_cur_state(simulate_type, dialog_history), 
                                                    word_vector=word_vector, 
                                                    dim_wordvec=dim_wordvec, 
                                                    n_encode_lstm_step=n_encode_lstm_step)

                generated_word_index, prob_logit = sess.run([caption_tf, probs], feed_dict={word_vectors: question})

                generated_sentence = generate_answer_sentence(generated_word_index=generated_word_index, 
                                                              prob_logit=prob_logit, 
                                                              ixtoword=ixtoword)

                dialog_history.append(generated_sentence)
                print('B => {}'.format(generated_sentence))

                question_2 = generate_question_vector(state=get_cur_state(simulate_type, dialog_history), 
                                                    word_vector=word_vector, 
                                                    dim_wordvec=dim_wordvec, 
                                                    n_encode_lstm_step=n_encode_lstm_step)

                generated_word_index, prob_logit = sess.run([caption_tf, probs], feed_dict={word_vectors: question_2})

                generated_sentence_2 = generate_answer_sentence(generated_word_index=generated_word_index, 
                                                                  prob_logit=prob_logit, 
                                                                  ixtoword=ixtoword)

                dialog_history.append(generated_sentence_2)
                print('A => {}'.format(generated_sentence_2))
                out.write('B: {}\nA: {}\n'.format(generated_sentence, generated_sentence_2))


if __name__ == "__main__":
    model_path = default_model_path if len(sys.argv) <= 1 else sys.argv[1]
    simulate_type = default_simulate_type if len(sys.argv) <= 2 else int(sys.argv[2])
    n_encode_lstm_step = n_encode_lstm_step * simulate_type + 1  # sent len * sent num + one random normal
    print('simulate_type', simulate_type)
    print('n_encode_lstm_step', n_encode_lstm_step)
    simulate(model_path=model_path, simulate_type=simulate_type)

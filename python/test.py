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
default_model_path = './model/Seq2Seq/model-77'
testing_data_path = 'sample_input.txt' if len(sys.argv) <= 2 else sys.argv[2]
output_path = 'sample_output_S2S.txt' if len(sys.argv) <= 3 else sys.argv[3]

word_count_threshold = config.WC_threshold

#=====================================================
# Train Parameters
#=====================================================
dim_wordvec = 300
dim_hidden = 1000

n_encode_lstm_step = 22 + 1 # one random normal as the first timestep
n_decode_lstm_step = 22

batch_size = 1

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

def test(model_path=default_model_path):
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
        print('\n=== Use model', model_path, '===\n')
        saver.restore(sess, model_path)
    except:
        print('\nUse default model\n')
        saver.restore(sess, default_model_path)

    with open(output_path, 'w') as out:
        generated_sentences = []
        bleu_score_avg = [0., 0.]
        for idx, question in enumerate(testing_data):
            print('question =>', question)

            question = [refine(w) for w in question.lower().split()]
            question = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in question]
            question.insert(0, np.random.normal(size=(dim_wordvec,))) # insert random normal at the first step

            if len(question) > n_encode_lstm_step:
                question = question[:n_encode_lstm_step]
            else:
                for _ in range(len(question), n_encode_lstm_step):
                    question.append(np.zeros(dim_wordvec))

            question = np.array([question]) # 1x22x300
    
            generated_word_index, prob_logit = sess.run([caption_tf, probs], feed_dict={word_vectors: question})
            
            # remove <unk> to second high prob. word
            for i in range(len(generated_word_index)):
                if generated_word_index[i] == 3:
                    sort_prob_logit = sorted(prob_logit[i][0])
                    maxindex = np.where(prob_logit[i][0] == sort_prob_logit[-1])[0][0]
                    secmaxindex = np.where(prob_logit[i][0] == sort_prob_logit[-2])[0][0]
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
            generated_sentence = generated_sentence.replace("i'll", "I'll")
            generated_sentence = generated_sentence.replace("i'v", "I'v")
            generated_sentence = generated_sentence.replace(" - ", "")

            print('generated_sentence =>', generated_sentence)
            out.write(generated_sentence + '\n')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test(model_path=sys.argv[1])
    else:
        test()

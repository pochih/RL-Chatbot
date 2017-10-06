#-*- coding: utf-8 -*-

from __future__ import print_function

import re
import os
import time
import sys

sys.path.append("../python")
import data_parser
import config

from gensim.models import KeyedVectors
from rl_model import PolicyGradient_chatbot
import tensorflow as tf
import numpy as np

#=====================================================
# Global Parameters
#=====================================================
default_model_path = './model/RL/model-56-3000'
default_simulate_type = 1  # type 1 use one former sent, type 2 use two former sents

testing_data_path = 'sample_input.txt' if len(sys.argv) <= 3 else sys.argv[3]
output_path = 'sample_dialog_output.txt' if len(sys.argv) <= 4 else sys.argv[4]

word_count_threshold = config.WC_threshold

word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
_, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)

#=====================================================
# Inference Parameters
#=====================================================
dim_wordvec = 300
dim_hidden = 1000

n_encode_lstm_step = 22 * 2 + 1 # one random normal as the first timestep
n_decode_lstm_step = 22

batch_size = 1

model = PolicyGradient_chatbot(
          dim_wordvec=dim_wordvec,
          n_words=len(ixtoword),
          dim_hidden=dim_hidden,
          batch_size=batch_size,
          n_encode_lstm_step=n_encode_lstm_step,
          n_decode_lstm_step=n_decode_lstm_step,
          bias_init_vector=bias_init_vector)

word_vectors, caption_tf, probs = model.build_generator()

sess = tf.InteractiveSession()

saver = tf.train.Saver()
try:
    print('\nUse default model\n')
    saver.restore(sess, default_model_path)
except:
    print('\nload model failed\n')

#=====================================================
# User Parameters
#=====================================================
user_list = []
user_timestamp = {}
user_dialog = {}


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

def simulate(uid, sent, model_path=default_model_path, simulate_type=default_simulate_type, n_encode_lstm_step=n_encode_lstm_step):
    ''' args:
            uid:            <type 'int'> user ID
            model_path:     <type 'str'> the pre-trained model using for inference
            simulate_type:  <type 'int'> how many former sents should use as state
    '''

    if dialog_history[uid] == []:
        simulate_type = 1
        n_encode_lstm_step = 22 + 1
        #dialog_history[uid] = init_history(simulate_type, sent)
        dialog_history[uid] = [sent]

    question = generate_question_vector(state=get_cur_state(simulate_type, dialog_history[uid]), 
                                        word_vector=word_vector, 
                                        dim_wordvec=dim_wordvec, 
                                        n_encode_lstm_step=n_encode_lstm_step)

    generated_word_index, prob_logit = sess.run([caption_tf, probs], feed_dict={word_vectors: question})

    generated_sentence = generate_answer_sentence(generated_word_index=generated_word_index, 
                                                  prob_logit=prob_logit, 
                                                  ixtoword=ixtoword)

    dialog_history[uid].append(generated_sentence)


@app.route("/")
def index():
    return redirect('rl-bot')

@app.route("/rl-bot")
def chatbot():
    # check if any use idle for 30 mins
    checkIdle()

    uid = random.randint(1, 1000)
    print('uid', uid)

    while uid in user_list:
        uid = random.randint(1, 1000)

    checkAndInitUID(uid)

    return render_template('RL_chatbot.html', uid=uid)

@app.route("/messages")
@app.route("/messages/<uid>")
def getMessage(uid=None):
    global user_dialog
    uid = int(uid)
    if uid != None and uid in user_dialog:
        res = ''

        if len(user_dialog[uid]) > 0:
            res = user_dialog[uid][-1]

        print('GET /messages res', res)
        return str(res)
    else:
        return str(user_dialog)

@app.route('/messages/<uid>', methods=['POST'])
def postMessage(uid=None):
    global user_dialog
    uid = int(uid)
    if request.method == 'POST' and uid != None:
        # check uid
        checkUID(uid)

        # parse user msg
        data = request.data.decode("utf-8") 
        print(type(data), data)
        data = data.strip('\x08\n\r\t ')
        req = ast.literal_eval(data)
        print('req', req)
        userMsg = req['msg']
        simulate(uid=uid, sent=userMsg)

    return "POST OK"

@app.route('/reset-msg/<uid>')
def resetUserMsg(uid=None):
    global user_dialog
    uid = int(uid)
    if uid != None:
        user_dialog[uid] = []
        user_timestamp[uid] = time.time()
    return "reset OK"

def checkUID(uid):
    if uid not in user_list:
        user_list.append(uid)
    if uid not in user_dialog:
        user_dialog[uid] = []
    if uid not in user_timestamp:
        user_timestamp[uid] = time.time()

def checkAndInitUID(uid):
    if uid not in user_list:
        user_list.append(uid)
    if uid not in user_dialog:
        user_dialog[uid] = []
    if uid not in user_timestamp:
        user_timestamp[uid] = time.time()

def checkIdle():
    cur_time = time.time()
    idle_user = []

    for uid in user_list:
        # idle for 30 mins
        if float(cur_time) - float(user_timestamp[uid]) > 1800:
            idle_user.append(uid)

    for uid in idle_user:
        user_list.remove(uid)
        user_dialog.pop(uid, None)
        user_timestamp.pop(uid, None)
        print("Remove user {}".format(uid))


if __name__ == "__main__":
    app.run(host=140.112.29.42, port=6666, debug=True)

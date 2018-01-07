# coding=utf-8

from __future__ import print_function
import cPickle as pickle
import time
import re
import numpy as np
from gensim.models import word2vec, KeyedVectors

WORD_VECTOR_SIZE = 300

raw_movie_conversations = open('data/movie_conversations.txt', 'r').read().split('\n')[:-1]

utterance_dict = pickle.load(open('data/utterance_dict', 'rb'))

ts = time.time()
corpus = word2vec.Text8Corpus("data/tokenized_all_words.txt")
word_vector = word2vec.Word2Vec(corpus, size=WORD_VECTOR_SIZE)
word_vector.wv.save_word2vec_format(u"model/word_vector.bin", binary=True)
word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

ts = time.time()
conversations = []
print('len conversation', len(raw_movie_conversations))
con_count = 0
traindata_count = 0
for conversation in raw_movie_conversations:
    conversation = conversation.split(' +++$+++ ')[-1]
    conversation = conversation.replace('[', '')
    conversation = conversation.replace(']', '')
    conversation = conversation.replace('\'', '')
    conversation = conversation.split(', ')
    assert len(conversation) > 1
    for i in range(len(conversation)-1):
        con_a = utterance_dict[conversation[i+1]].strip()
        con_b = utterance_dict[conversation[i]].strip()
        if len(con_a.split()) <= 22 and len(con_b.split()) <= 22:
            con_a = [refine(w) for w in con_a.lower().split()]
            # con_a = [word_vector[w] if w in word_vector else np.zeros(WORD_VECTOR_SIZE) for w in con_a]
            conversations.append((con_a, con_b))
            traindata_count += 1
    con_count += 1
    if con_count % 1000 == 0:
        print('con_count {}, traindata_count {}'.format(con_count, traindata_count))
pickle.dump(conversations, open('data/reversed_conversations_lenmax22', 'wb'), True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))

# some statistics of training data
max_a = -1
max_b = -1
max_a_ind = -1
max_b_ind = -1
sum_a = 0.
sum_b = 0.
len_a_list = []
len_b_list = []
for i in range(len(conversations)):
    len_a = len(conversations[i][0])
    len_b = len(conversations[i][1].split())
    if len_a > max_a:
        max_a = len_a
        max_a_ind = i
    if len_b > max_b:
        max_b = len_b
        max_b_ind = i
    sum_a += len_a
    sum_b += len_b
    len_a_list.append(len_a)
    len_b_list.append(len_b)
np.save("data/reversed_lenmax22_a_list", np.array(len_a_list))
np.save("data/reversed_lenmax22_b_list", np.array(len_b_list))
print("max_a_ind {}, max_b_ind {}".format(max_a_ind, max_b_ind))
print("max_a {}, max_b {}, avg_a {}, avg_b {}".format(max_a, max_b, sum_a/len(conversations), sum_b/len(conversations)))

ts = time.time()
conversations = []
# former_sents = []
print('len conversation', len(raw_movie_conversations))
con_count = 0
traindata_count = 0
for conversation in raw_movie_conversations:
    conversation = conversation.split(' +++$+++ ')[-1]
    conversation = conversation.replace('[', '')
    conversation = conversation.replace(']', '')
    conversation = conversation.replace('\'', '')
    conversation = conversation.split(', ')
    assert len(conversation) > 1
    con_a_1 = ''
    for i in range(len(conversation)-1):
        con_a_2 = utterance_dict[conversation[i]]
        con_b = utterance_dict[conversation[i+1]]
        if len(con_a_1.split()) <= 22 and len(con_a_2.split()) <= 22 and len(con_b.split()) <= 22:
            con_a = "{} {}".format(con_a_1, con_a_2)
            con_a = [refine(w) for w in con_a.lower().split()]
            # con_a = [word_vector[w] if w in word_vector else np.zeros(WORD_VECTOR_SIZE) for w in con_a]
            conversations.append((con_a, con_b, con_a_2))
            # former_sents.append(con_a_2)
            traindata_count += 1
        con_a_1 = con_a_2
    con_count += 1
    if con_count % 1000 == 0:
        print('con_count {}, traindata_count {}'.format(con_count, traindata_count))
pickle.dump(conversations, open('data/conversations_lenmax22_formersents2_with_former', 'wb'), True)
# pickle.dump(former_sents, open('data/conversations_lenmax22_former_sents', 'wb'), True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))

ts = time.time()
conversations = []
# former_sents = []
print('len conversation', len(raw_movie_conversations))
con_count = 0
traindata_count = 0
for conversation in raw_movie_conversations:
    conversation = conversation.split(' +++$+++ ')[-1]
    conversation = conversation.replace('[', '')
    conversation = conversation.replace(']', '')
    conversation = conversation.replace('\'', '')
    conversation = conversation.split(', ')
    assert len(conversation) > 1
    con_a_1 = ''
    for i in range(len(conversation)-1):
        con_a_2 = utterance_dict[conversation[i]]
        con_b = utterance_dict[conversation[i+1]]
        if len(con_a_1.split()) <= 22 and len(con_a_2.split()) <= 22 and len(con_b.split()) <= 22:
            con_a = "{} {}".format(con_a_1, con_a_2)
            con_a = [refine(w) for w in con_a.lower().split()]
            # con_a = [word_vector[w] if w in word_vector else np.zeros(WORD_VECTOR_SIZE) for w in con_a]
            conversations.append((con_a, con_b))
            # former_sents.append(con_a_2)
            traindata_count += 1
        con_a_1 = con_a_2
    con_count += 1
    if con_count % 1000 == 0:
        print('con_count {}, traindata_count {}'.format(con_count, traindata_count))
pickle.dump(conversations, open('data/conversations_lenmax22_former_sents2', 'wb'), True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))

ts = time.time()
conversations = []
print('len conversation', len(raw_movie_conversations))
con_count = 0
traindata_count = 0
for conversation in raw_movie_conversations:
    conversation = conversation.split(' +++$+++ ')[-1]
    conversation = conversation.replace('[', '')
    conversation = conversation.replace(']', '')
    conversation = conversation.replace('\'', '')
    conversation = conversation.split(', ')
    assert len(conversation) > 1
    for i in range(len(conversation)-1):
        con_a = utterance_dict[conversation[i]]
        con_b = utterance_dict[conversation[i+1]]
        if len(con_a.split()) <= 22 and len(con_b.split()) <= 22:
            con_a = [refine(w) for w in con_a.lower().split()]
            # con_a = [word_vector[w] if w in word_vector else np.zeros(WORD_VECTOR_SIZE) for w in con_a]
            conversations.append((con_a, con_b))
            traindata_count += 1
    con_count += 1
    if con_count % 1000 == 0:
        print('con_count {}, traindata_count {}'.format(con_count, traindata_count))
pickle.dump(conversations, open('data/conversations_lenmax22', 'wb'), True)
print("Time Elapsed: {} secs\n".format(time.time() - ts))

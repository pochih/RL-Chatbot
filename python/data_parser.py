# coding=utf-8

from __future__ import print_function
import pickle
import codecs
import re
import os
import time
import numpy as np
import config

def preProBuildWordVocab(word_count_threshold=5, all_words_path=config.all_words_path):
    # borrowed this function from NeuralTalk

    if not os.path.exists(all_words_path):
        parse_all_words(all_words_path)

    corpus = open(all_words_path, 'r').read().split('\n')[:-1]
    captions = np.asarray(corpus, dtype=np.object)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in captions:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def parse_all_words(all_words_path):
    raw_movie_lines = open('data/movie_lines.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')[:-1]

    with codecs.open(all_words_path, "w", encoding='utf-8', errors='ignore') as f:
        for line in raw_movie_lines:
            line = line.split(' +++$+++ ')
            utterance = line[-1]
            f.write(utterance + '\n')

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

if __name__ == '__main__':
    parse_all_words(config.all_words_path)

    raw_movie_lines = open('data/movie_lines.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')[:-1]
    
    utterance_dict = {}
    with codecs.open('data/tokenized_all_words.txt', "w", encoding='utf-8', errors='ignore') as f:
        for line in raw_movie_lines:
            line = line.split(' +++$+++ ')
            line_ID = line[0]
            utterance = line[-1]
            utterance_dict[line_ID] = utterance
            utterance = " ".join([refine(w) for w in utterance.lower().split()])
            f.write(utterance + '\n')
    pickle.dump(utterance_dict, open('data/utterance_dict', 'wb'), True)

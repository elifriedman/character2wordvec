
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec
from tensorflow.nn.rnn_cell import GRUCell, LSTMCell

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

def model(max_length, dim, num_hidden):
    sequence = tf.placeholder(tf.float32, [None, max_length, dim])
    length = length(sequence)
    output, state = tf.nn.dynamic_rnn(
        LSTMCell(num_hidden),
        sequence,
        dtype=tf.float32,
        sequence_length=length
    )
    return last_relevant(output, length)


def make_dataset(filename, session):
    (words, counts, words_per_epoch, epoch, num_processed, examples,
     labels) = word2vec.skipgram(filename=filename,
                                 batch_size=32,
                                 window_size=5,
                                 min_count=1,
                                 subsample=0.001)
    (vocab_words, vocab_counts,
     words_per_epoch) = session.run([words, counts, words_per_epoch])
    vocab_size = len(vocab_words)
    id2word = vocab_words
    word2id = {}
    for i, w in enumerate(id2word):
        word2id[w] = i
    return (id2word, word2id, vocab_counts)
      

def main(_):
    if len(sys.argv) < 2:
        print("usage: " + sys.argv[0] + " <filename>")
        sys.exit(-1)
    filename = sys.argv[1]
    session = tf.Session()
    id2word, word2id, counts = make_dataset(filename, session)
    print(id2word)

if __name__ == "__main__":
  tf.app.run()

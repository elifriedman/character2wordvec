
import tensorflow as tf

word2idx = tf.load_op_library("word2idx/word_to_idx_op_kernel.so")
NUM_CHARACTERS = 94 # number of useful ASCII characters

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

def recurrent_model(words, max_wordlen, rnn_cell, output_last=True):
    character_idxs = word2idx.word_to_idx(words, max_wordlen)
    character_idxs.set_shape([None, max_wordlen])
    chars_onehot = tf.one_hot(character_idxs, NUM_CHARACTERS)
    seqlen = length(chars_onehot)
    output, state = tf.nn.dynamic_rnn(
        rnn_cell,
        chars_onehot,
        dtype=tf.float32,
        sequence_length=seqlen,
    )
    return last_relevant(output, seqlen) if output_last else output

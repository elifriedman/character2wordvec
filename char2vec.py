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
import char_rnn

from tensorflow.models.embedding import gen_word2vec as word2vec

flags = tf.app.flags

flags.DEFINE_string("save_path", "output/", "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data", "data/text8_small", "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 512, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("num_threads", 8,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 1,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.
    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.num_threads = FLAGS.num_threads

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # How often to print statistics.
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file (rounds up to the nearest
    # statistics_interval).
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints (rounds up to the nearest statistics
    # interval).
    self.checkpoint_interval = FLAGS.checkpoint_interval

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    # Eval options.
    # The text file for eval.
    self.eval_data = FLAGS.eval_data


class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph()
    session.run(tf.initialize_all_variables())
    self.saver = tf.train.Saver()

  def build_graph(self):
    """Build the graph for the full model."""
    opts = self._options
    # The training data. A text file.
    (words, counts, words_per_epoch, self._epoch, self._words, examples,
     labels) = word2vec.skipgram(filename=opts.train_data,
                                 batch_size=opts.batch_size,
                                 window_size=opts.window_size,
                                 min_count=opts.min_count,
                                 subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    max_wordlen = 0
    max_wordlen = reduce(lambda cur_max_wordlen, val: max(cur_max_wordlen, len(val)), opts.vocab_words, max_wordlen)

    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)
    self._examples = examples
    self._labels = labels
    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i
    self.true_logits, self.sampled_logits = self.forward(examples, labels, words, max_wordlen)
    loss = self.nce_loss(self.true_logits, self.sampled_logits)
    tf.scalar_summary("NCE loss", loss)
    self._loss = loss
    self.optimize(loss)

    

  def forward(self, examples, labels, words, max_wordlen):
    """Build the graph for the forward pass."""
    opts = self._options

    # Global step: scalar, i.e., shape [].
    self.global_step = tf.Variable(0, name="global_step")

    # Nodes to compute the nce loss w/ candidate sampling.
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [opts.batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=opts.num_samples,
        unique=True,
        range_max=opts.vocab_size,
        distortion=0.75,
        unigrams=opts.vocab_counts.tolist()))

    example_words = tf.gather(words, examples)
    label_words = tf.gather(words, labels)
    sampled_words = tf.gather(words, sampled_ids)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(opts.emb_dim) # I think I can reuse this
    lstm_bias = tf.nn.rnn_cell.LSTMCell(1)

    # Embeddings for examples: [batch_size, emb_dim]
    with tf.variable_scope("words"):
        example_emb = char_rnn.recurrent_model(example_words, max_wordlen, lstm_cell)

    # Weights for labels: [batch_size, emb_dim]
    with tf.variable_scope("contexts"):
        labels_emb = char_rnn.recurrent_model(label_words, max_wordlen, lstm_cell)
    with tf.variable_scope("contexts_b"):
        labels_bias = char_rnn.recurrent_model(label_words, max_wordlen, lstm_bias)

    # Weights for sampled ids: [num_sampled, emb_dim]
    with tf.variable_scope("contexts", reuse=True):
        sampled_emb = char_rnn.recurrent_model(sampled_words, max_wordlen, lstm_cell)
    with tf.variable_scope("contexts_b", reuse=True):
        sampled_bias = char_rnn.recurrent_model(sampled_words, max_wordlen, lstm_bias)

    # True logits: [batch_size, 1]
    labels_b_vec = tf.reshape(labels_bias, [opts.batch_size])
    true_logits = tf.reduce_sum(tf.mul(example_emb, labels_emb), 1) + labels_b_vec

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.
    sampled_b_vec = tf.reshape(sampled_bias, [opts.num_samples])
    sampled_logits = tf.matmul(example_emb,
                               sampled_emb,
                               transpose_b=True) + sampled_b_vec
    return true_logits, sampled_logits

  def nce_loss(self, true_logits, sampled_logits):
    """Build the graph for the NCE loss."""

    # cross-entropy(logits, labels)
    opts = self._options
    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        true_logits, tf.ones_like(true_logits))
    sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        sampled_logits, tf.zeros_like(sampled_logits))

    # NCE-loss is the sum of the true and noise (sampled words)
    # contributions, averaged over the batch.
    nce_loss_tensor = (tf.reduce_sum(true_xent) +
                       tf.reduce_sum(sampled_xent)) / opts.batch_size
    return nce_loss_tensor

  def optimize(self, loss):
    """Build the graph to optimize the loss function."""

    # Optimizer nodes.
    # Linear learning rate decay.
    opts = self._options
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
    self._lr = lr
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss,
                               global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
    self._train = train

   def _train_thread_body(self):
       initial_epoch, = self._session.run([self._epoch])
       epoch = initial_epoch
       while epoch == initial_epoch:
           _, epoch = self._session.run([self._train, self._epoch])

  def train(self, num_epochs):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in xrange(opts.num_threads):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time, last_summary_time = initial_words, time.time(), 0
    last_checkpoint_time = 0
    epoch = initial_epoch
    while epoch < initial_epoch + num_epochs: # run for num_epochs
        time.sleep(opts.statistics_interval)  # Reports our progress once a while.
        (epoch, step, loss, words, lr) = self._session.run(
            [self._epoch, self.global_step, self._loss, self._words, self._lr])
        now = time.time()
        last_words, last_time, rate = words, now, (words - last_words) / (
            now - last_time)
        print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
              (epoch, step, lr, loss, rate), end="")
        sys.stdout.flush()
        if now - last_checkpoint_time > opts.checkpoint_interval:
            self.saver.save(self._session,
                            os.path.join(opts.save_path, "model.ckpt"),
                            global_step=step.astype(int))
            last_checkpoint_time = now

    for t in workers:
      t.join()

    return epoch
   



def init():
    opts = Options()
    sess = tf.Session()
    model = Word2Vec(opts, sess)
    return opts, sess, model

from __future__ import print_function
from __future__ import division
from enum import Enum
from tqdm import tqdm, trange
import numpy as np
import tensorflow as tf
import itertools
import dataset_utils as data_utils
import sys
import codecs
import hyper_params
import utils
import eval_utils
import model


class NMTModel:
    def __init__(self, params):
        self.params = params
        self.checkpoints_path = self.params.checkpoints_path
        (self.src_int2word, self.tgt_int2word,
         self.src_word2int, self.tgt_word2int) = data_utils.get_dicts(
            self.params.src_vocab_filename, self.params.tgt_vocab_filename)
        self.test_src_size = data_utils.get_vocab_size(
            self.params.test_src_filename)
        self.test_tgt_size = data_utils.get_vocab_size(
            self.params.test_tgt_filename)
        self.train_size = data_utils.get_vocab_size(
            self.params.src_filename)
        self.train_size -= 150  # workaround for vietnamese having empty lines

        # creating training graph
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # iterator
            self.train_iterator = data_utils.get_train_dataset(
                self.params.src_filename, self.params.tgt_filename,
                self.params.src_vocab_filename, self.params.tgt_vocab_filename,
                batch_size=self.params.batch_size,
                src_max_len=self.params.src_max_len,
                tgt_max_len=self.params.tgt_max_len)
            # updating hparams and getting some ids
            hyper_params.update_hparams(
                self.params,
                src_vocab_size=data_utils.get_vocab_size(
                    self.params.src_vocab_filename),
                tgt_vocab_size=data_utils.get_vocab_size(
                    self.params.tgt_vocab_filename))
            # building model
            self.train_model = model._NMTModel(
                self.train_iterator, self.params, model.Mode.TRAIN,
                self.train_graph)
            self.train_initializer = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()

        # creating inference graph
        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default():
            self.infer_filename_placeholder = tf.placeholder(tf.string)
            # iterator
            self.infer_iterator = data_utils.get_infer_dataset(
                self.infer_filename_placeholder,
                self.params.src_vocab_filename,
                self.params.infer_batch_size)
            # building model
            self.infer_model = model._NMTModel(
                self.infer_iterator, self.params, model.Mode.INFER,
                self.infer_graph)
            self.infer_saver = tf.train.Saver()

        self._make_sessions_and_initialize()
        utils.print_time('Created model with following parameters:')
        for k in ['name',
                  'cell',
                  'num_units',
                  'num_layers',
                  'embeddings_size',
                  'dropout',
                  'optimizer',
                  'learning_rate',
                  'decay_factor',
                  'start_decay_step',
                  'decay_steps',
                  'batch_size',
                  'infer_batch_size',
                  'bidirectional_encoder',
                  'bi_reduce',
                  'attention',
                  'infer_helper',
                  'time_major',
                  'max_gradient_norm',
                  'src_max_len',
                  'tgt_max_len',
                  'checkpoints_path',
                  'src_filename',
                  'tgt_filename',
                  'src_vocab_filename',
                  'tgt_vocab_filename',
                  'src_vocab_size',
                  'tgt_vocab_size',
                  'test_src_filename',
                  'test_tgt_filename']:
            print('\t%s:' % k, self.params[k])

    def _make_sessions_and_initialize(self):
        self.train_sess = tf.Session(graph=self.train_graph)
        # initializing tables, iterator and variables
        self.train_sess.run(self.train_iterator.tables_initializer)
        self.train_sess.run(self.train_iterator.initializer)
        self.train_sess.run(self.train_initializer)

        self.infer_sess = tf.Session(graph=self.infer_graph)
        self.infer_sess.run(self.infer_iterator.tables_initializer)

    def eval(self):
        # to shorten the notation
        params = self.params
        self.checkpoint_path = self.train_saver.save(
            self.train_sess, self.checkpoints_path)
        feed_dict = {
            self.infer_filename_placeholder: params.test_src_filename}
        self.infer_sess.run(
            self.infer_iterator.initializer, feed_dict=feed_dict)
        self.infer_saver.restore(self.infer_sess, self.checkpoint_path)

        num_batches = int(self.test_src_size / params.infer_batch_size)
        translations = []
        for i in trange(num_batches, desc='Evaluating', leave=False):
            logits = self.infer_sess.run(self.infer_model.logits)
            translations.append(logits.argmax(axis=2))

        translations_str = []
        for translations_batch in translations:
            for sentence in translations_batch:
                sentence = [self.tgt_int2word[w] for w in sentence]
                sentence = [w for w in itertools.takewhile(
                    lambda s: s != '</s>', sentence)]
                # sentence = [('' if w == '</s>' else w) for w in sentence]
                translations_str.append(' '.join(sentence))
        with codecs.open(params.eval_filename, 'w', encoding='utf-8') as f:
            f.write('')
            for line in translations_str:
                f.write(line + '\n')
        b1 = eval_utils.bleu(params.test_tgt_filename, params.eval_filename)

        translations_str = []
        for translations_batch in translations:
            for sentence in translations_batch:
                sentence = [
                    self.tgt_int2word[w]
                    for i, w in enumerate(sentence)
                    if i == 0 or sentence[i - 1] != sentence[i]]
                sentence = [w for w in itertools.takewhile(
                    lambda s: s != '</s>', sentence)]
                # sentence = [('' if w == '</s>' else w) for w in sentence]
                translations_str.append(' '.join(sentence))
        with codecs.open(params.eval_filename + '2', 'w',
                         encoding='utf-8') as f:
            f.write('')
            for line in translations_str:
                f.write(line + '\n')
        b2 = eval_utils.bleu(
            params.test_tgt_filename, params.eval_filename + '2')

        return b1, b2

    def train(self, num_epochs, eval_every, train_part=1.0):
        utils.print_time('Starting training')
        b1, b2 = 0.0
        for epoch in range(num_epochs):
            self.train_sess.run(self.train_model.iterator.initializer)
            num_batches = int(
                self.train_size / self.params.batch_size * train_part)
            # num_batches = 2
            tr = trange(num_batches, leave=True)
            tr.set_description('E %d/%d' % (epoch + 1, num_epochs))
            losses_sum = 0.0
            perp_sum = 0.0
            for step in tr:
                _, loss, perplexity, lrate = self.train_sess.run(
                    [self.train_model.train_op, self.train_model.loss,
                     self.train_model.perplexity,
                     self.train_model.learning_rate])
                losses_sum += loss
                perp_sum += perplexity
                if (step + 1) % eval_every == 0:
                    b1, b2 = self.eval()
                tr.set_postfix(
                    l=loss, bleu=b1, b2=b2, p=perplexity, lrate=lrate)
            b1, b2 = self.eval()
            utils.print_time(
                'After epoch %d/%d, bleu: %.4f, b2: %.4f, avg loss: %.4f, '
                'avg perplexity: %.4f' % (
                    epoch + 1, num_epochs, b1, b2, losses_sum / num_batches,
                    perp_sum / num_batches))
            sys.stdout.flush()

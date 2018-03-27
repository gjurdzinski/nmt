from __future__ import print_function
from __future__ import division
import collections
import codecs
import tensorflow as tf
import hyper_params


UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_ID = 0


class BatchedInput(
    collections.namedtuple('BatchedInput',
                           ('initializer', 'source', 'target_input',
                            'target_output', 'source_sequence_length',
                            'target_sequence_length', 'tables_initializer'))):
    pass


# tf.tables_initializer().run()
def _creat_vocab_table(vocab_filename):
    return tf.contrib.lookup.index_table_from_file(
        vocab_filename, default_value=UNK_ID)


def get_train_dataset(src_filename, tgt_filename,
                      src_vocab_filename, tgt_vocab_filename,
                      batch_size, output_buffer_size=None,
                      src_max_len=None, tgt_max_len=None):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    # creating dataset objects
    src_dataset = tf.data.TextLineDataset(src_filename)
    tgt_dataset = tf.data.TextLineDataset(tgt_filename)
    # reading vocabs to tables
    src_vocab = _creat_vocab_table(src_vocab_filename)
    tgt_vocab = _creat_vocab_table(tgt_vocab_filename)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, seed=42)

    src_eos_id = tf.cast(src_vocab.lookup(tf.constant(EOS)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab.lookup(tf.constant(SOS)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab.lookup(tf.constant(EOS)), tf.int32)

    # splitting sentences into arrays of words
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.string_split([src]).values,
                          tf.string_split([tgt]).values)).prefetch(
        output_buffer_size)
    # replacing words with their ids
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab.lookup(src), tf.int32),
                          tf.cast(tgt_vocab.lookup(tgt), tf.int32))).prefetch(
        output_buffer_size)

    # filtering empty sequences
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0,
                                        tf.size(tgt) > 0)).prefetch(
        output_buffer_size)

    # limiting sentences length
    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt)).prefetch(
            output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len])).prefetch(
            output_buffer_size)

    # create a tgt_input prefixed with SOS and a tgt_output suffixed with EOS
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src,
                          tf.concat(([tgt_sos_id], tgt), 0),
                          tf.concat((tgt, [tgt_eos_id]), 0))).prefetch(
        output_buffer_size)
    # add sequence lengths
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out,
                                      tf.size(src), tf.size(tgt_in))).prefetch(
        output_buffer_size)

    # batching input
    src_eos_id = tf.cast(src_vocab.lookup(tf.constant(EOS)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab.lookup(tf.constant(EOS)), tf.int32)
    batched_dataset = src_tgt_dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),      # src_len
            tf.TensorShape([])),     # tgt_len
        # pad the source and target sequences with eos tokens
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,           # src_len -- unused
            0))          # tgt_len -- unused

    iterator = batched_dataset.make_initializable_iterator()
    # session.run(batched_iterator.initializer, feed_dict={...})
    (src_ids, tgt_in_ids, tgt_out_ids, src_seq_len,
     tgt_seq_len) = (iterator.get_next())
    return BatchedInput(
        initializer=iterator.initializer,
        source=src_ids,
        target_input=tgt_in_ids,
        target_output=tgt_out_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len,
        tables_initializer=tf.tables_initializer())


def get_infer_dataset(filename, vocab_filename, batch_size):
    src_dataset = tf.data.TextLineDataset(filename)
    src_vocab = _creat_vocab_table(vocab_filename)
    src_eos_id = tf.cast(src_vocab.lookup(tf.constant(EOS)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
    src_dataset = src_dataset.map(
        lambda src: tf.cast(src_vocab.lookup(src), tf.int32))
    src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

    batched_dataset = src_dataset.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),     # src_len
        padding_values=(
            src_eos_id,  # src
            0))          # src_len -- unused

    iterator = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = iterator.get_next()
    return BatchedInput(
        initializer=iterator.initializer,
        source=src_ids,
        target_input=None,
        target_output=None,
        source_sequence_length=src_seq_len,
        target_sequence_length=None,
        tables_initializer=tf.tables_initializer())


def get_vocab_size(vocab_filename):
    return len(open(vocab_filename, 'r').readlines())


def get_dicts(src_vocab_filename, tgt_vocab_filename):
    with codecs.open(src_vocab_filename, 'r', encoding='utf-8') as f:
        src_int2word = f.read().splitlines()
    with codecs.open(tgt_vocab_filename, 'r', encoding='utf-8') as f:
        tgt_int2word = f.read().splitlines()
    src_word2int = dict(zip(src_int2word, range(len(src_int2word))))
    tgt_word2int = dict(zip(tgt_int2word, range(len(tgt_int2word))))
    return src_int2word, tgt_int2word, src_word2int, tgt_word2int

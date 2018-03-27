from __future__ import print_function
from __future__ import division
from enum import Enum
import tensorflow as tf
import hyper_params
import utils
import dataset_utils as data_utils


def _get_cell(cell_str, num_units, dropout):
    if cell_str == 'LSTM':
        cell_type = tf.contrib.rnn.BasicLSTMCell
    else:
        raise ValueError('Unsupported cell type: %s' % cell_str)
    cell = cell_type(num_units)
    if dropout is not None and dropout > 0.0:
        # I've always done it a bit different, now doing as in tutorial
        cell = tf.contrib.rnn.DropoutWrapper(
            cell=cell, input_keep_prob=1.0-dropout, dtype=tf.float32)
    # TODO: residual
    return cell


def _get_multi_layer_cell(cell_str, num_units, num_layers, dropout):
    return tf.contrib.rnn.MultiRNNCell(
        [_get_cell(cell_str, num_units, dropout) for _ in range(num_layers)]
    )


def _get_optimizer(name, learning_rate):
    if name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif name == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type: %s' % name)


class Mode(Enum):
    TRAIN = 1
    EVAL = 2
    INFER = 3


class _NMTModel:
    def __init__(self, iterator, params, mode, graph):
        """
        Arguments:
            iterator: iterator to the data, see data_utils.py, should be
                created within graph argument
            params: HParams object (see hyper_params.py)
            mode: Mode.TRAIN | Mode.EVAL | Model.INFER
            graph: tf.Graph
        """
        self.params = params
        self.iterator = iterator
        self.mode = mode
        self.graph = graph
        with self.graph.as_default(), tf.variable_scope(self.params.name):
            self._build_graph()
            self.saver = tf.train.Saver()

    def _get_learning_rate(self):
        params = self.params
        self.learning_rate = tf.constant(params.learning_rate)
        if params.decay_factor is not None:
            return tf.cond(
                self.global_step < params.start_decay_step,
                lambda: self.learning_rate,
                lambda: tf.train.exponential_decay(
                    self.learning_rate,
                    (self.global_step - params.start_decay_step),
                    params.decay_steps, params.decay_factor, staircase=True),
                name="lrate_decay_cond")
        else:
            return self.learning_rate

    def _get_max_time(self, tensor):
        time_axis = 0 if self.params.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    def _get_batch_size(self, tensor):
        batch_axis = 1 if self.params.time_major else 0
        return tensor.shape[batch_axis].value or tf.shape(tensor)[batch_axis]


    def _get_loss(self, logits):
        target_output = self.iterator.target_output
        if self.params.time_major:
            target_output = tf.transpose(target_output)
        max_time = self._get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        # masks sequences lengths
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time,
            dtype=self.logits.dtype)
        if self.params.time_major:
            target_weights = tf.transpose(target_weights)
        loss = tf.reduce_sum(
            crossent * target_weights) / tf.cast(
            self._get_batch_size(target_output), tf.float32)
        return loss

    def _get_infer_max_iter(self, source_sequence_length):
        factor = 2.0
        max_src_len = tf.reduce_max(source_sequence_length)
        return tf.to_int32(tf.round(tf.to_float(max_src_len) * factor))

    def _get_encoder(self):
        params = self.params
        iterator = self.iterator
        # creating encoder embeddings
        encoder_embeddings = tf.get_variable(
            'encoder_embeddings',
            shape=[params.src_vocab_size, params.embeddings_size],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        encoder_emb_input = tf.nn.embedding_lookup(
            encoder_embeddings, iterator.source)

        if params.bidirectional_encoder:
            if params.bi_reduce == 'layers':
                if params.num_layers % 2 != 0:
                    raise ValueError(
                        'When using bidirectional encoder with reducing layers'
                        ' the number of layers shoud be even, is '
                        '%d' % params.num_layers)
                num_bi_layers = int(params.num_layers / 2)
                num_bi_units = params.num_units
            elif params.bi_reduce == 'units':
                if params.num_units % 2 != 0:
                    raise ValueError(
                        'When using bidirectional encoder with reducing units '
                        'the number of units shoud be even, is '
                        '%d' % params.num_units)
                num_bi_layers = params.num_layers
                num_bi_units = int(params.num_units / 2)
            else:
                raise ValueError('Unsupported reduction for bidirectional '
                                 'encoder: %s' % str(params.bi_reduce))

            backward_encoder_cell = _get_multi_layer_cell(
                params.cell, num_bi_units, num_bi_layers,
                params.dropout if self.mode == Mode.TRAIN else None)
            forward_encoder_cell = _get_multi_layer_cell(
                params.cell, num_bi_units, num_bi_layers,
                params.dropout if self.mode == Mode.TRAIN else None)
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                forward_encoder_cell, backward_encoder_cell, encoder_emb_input,
                sequence_length=iterator.source_sequence_length,
                time_major=params.time_major, dtype=tf.float32)
            encoder_outputs = tf.concat(bi_outputs, -1)
            encoder_state = []
            for l_id in range(num_bi_layers):
                if params.bi_reduce == 'layers':
                    encoder_state.append(bi_state[0][l_id])  # forward
                    encoder_state.append(bi_state[1][l_id])  # backward
                else:
                    to_append = tf.contrib.rnn.LSTMStateTuple(
                        c=tf.concat(
                            [bi_state[0][l_id].c, bi_state[1][l_id].c], 1),
                        h=tf.concat(
                            [bi_state[0][l_id].h, bi_state[1][l_id].h], 1))
                    encoder_state.append(to_append)
            encoder_state = tuple(encoder_state)
        else:
            encoder_cell = _get_multi_layer_cell(
                params.cell, params.num_units, params.num_layers,
                params.dropout if self.mode == Mode.TRAIN else None)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_emb_input,
                sequence_length=iterator.source_sequence_length,
                time_major=params.time_major, dtype=tf.float32)
        return encoder_outputs, encoder_state

    def _get_decoder(self, use_att, encoder_out, encoder_state, decoder_scope):
        params = self.params
        iterator = self.iterator
        # creating decoder embeddings
        decoder_embeddings = tf.get_variable(
            'decoder_embeddings',
            shape=[params.tgt_vocab_size, params.embeddings_size],
            dtype=tf.float32, initializer=tf.random_uniform_initializer(
                -0.1, 0.1))

        # building decoder
        decoder_cell = _get_multi_layer_cell(
            params.cell, params.num_units, params.num_layers,
            params.dropout if self.mode == Mode.TRAIN else None)
        # projection layer
        projection_layer = tf.layers.Dense(
            params.tgt_vocab_size, use_bias=False,
            name='projection_layer')

        # constructing attention mechanism
        if use_att:
            if params.time_major:
                attention_states = tf.transpose(encoder_out, [1, 0, 2])
            else:
                attention_states = encoder_out

            # Create an attention mechanism
            if params.attention == 'luong':
                # memory_sequence_length -- zeroing encoder states past
                # these lengths
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    params.num_units, attention_states,
                    memory_sequence_length=iterator.source_sequence_length,
                    scale=False)
            else:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    params.num_units, attention_states,
                    memory_sequence_length=iterator.source_sequence_length,
                    normalize=False)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, attention_mechanism,
                attention_layer_size=params.num_units)

        if use_att:
            # attention needs state to be of type AttentionWrapperState
            decoder_initial_state = decoder_cell.zero_state(
                self._get_batch_size(iterator.source), tf.float32).clone(
                cell_state=encoder_state)
        else:
            decoder_initial_state = encoder_state

        if self.mode != Mode.INFER:  # TRAIN or EVAL mode
            decoder_emb_input = tf.nn.embedding_lookup(
                decoder_embeddings, iterator.target_input)
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_input, iterator.target_sequence_length,
                time_major=params.time_major)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, decoder_initial_state,
                output_layer=projection_layer if params.save_memory else None)

            # dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=params.time_major,
                swap_memory=True,
                scope=decoder_scope)
            logits = outputs.rnn_output
            if not params.save_memory:
                logits = projection_layer(logits)

        else:  # INFER mode
            start_tokens = tf.fill(
                [self._get_batch_size(iterator.source)], params.tgt_sos_id)
            end_token = params.tgt_eos_id
            if params.infer_helper == 'greedy':
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    decoder_embeddings, start_tokens, end_token)
            elif params.infer_helper == 'sample':
                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    decoder_embeddings, start_tokens, end_token)
            else:
                raise ValueError(
                    'Unknown infer helper: %s' % params.infer_helper)
            maximum_iterations = self._get_infer_max_iter(
                iterator.source_sequence_length)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, decoder_initial_state,
                output_layer=projection_layer)

            # dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=params.time_major,
                swap_memory=True,
                maximum_iterations=maximum_iterations,
                scope=decoder_scope)
            logits = outputs.rnn_output
        return logits

    def _build_graph(self):
        # copying these for shorter notation
        params = self.params
        iterator = self.iterator

        if params.attention in ['luong', 'bahdanau']:
            use_attention = True
        elif params.attention == 'none':
            use_attention = False
        else:
            raise ValueError('Unknown attention type: %s' % params.attention)

        with tf.variable_scope('encoder'):
            encoder_outputs, encoder_state = self._get_encoder()

        with tf.variable_scope('decoder') as decoder_scope:
            self.logits = self._get_decoder(
                use_attention, encoder_outputs, encoder_state, decoder_scope)

        if self.mode != Mode.INFER:
            self.loss = self._get_loss(self.logits)
            self.predict_count = tf.reduce_sum(iterator.target_sequence_length)
            self.perplexity = tf.exp(
                params.batch_size * self.loss / tf.cast(
                    self.predict_count, tf.float32),
                name='perplexity')

        if self.mode == Mode.TRAIN:
            self.global_step = tf.Variable(
                0, trainable=False, name='global_step')
            # optimizer
            self.learning_rate = self._get_learning_rate()
            self.optimizer = _get_optimizer(
                params.optimizer, self.learning_rate)
            if params.max_gradient_norm is not None:
                trainable_variables = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_variables)
                clipped_gradients, gradient_norm = tf.clip_by_global_norm(
                    gradients, clip_norm=params.max_gradient_norm)
                self.train_op = self.optimizer.apply_gradients(
                    zip(clipped_gradients, trainable_variables),
                    global_step=self.global_step)
            else:
                self.train_op = self.optimizer.minimize(
                    self.train_loss, global_step=self.global_step)

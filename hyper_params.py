from __future__ import print_function
from __future__ import division
import tensorflow as tf

# DATA_PATH = '/Users/grzegorz/Rozne/Datasets/wmt16_de_en/'
DATA_PATH = '/home/z1137405/Rozne/Datasets/wmt/'
DATA_PATH = '/home/z1137405/Rozne/Datasets/vi/'


class HParams(dict):
    def __init__(self, **kwargs):
        self.__dict__ = self
        self.update(kwargs)


def get_hparams(**kwargs):
    hparams = HParams(
        name='nmt',
        batch_size=32,
        infer_batch_size=16,
        embeddings_size=512,
        cell='LSTM',
        num_units=512,
        num_layers=4,
        dropout=0.2,
        bidirectional_encoder=True,
        bi_reduce='units',  # 'layers', 'units'
        attention='luong',  # 'luong', 'bahdanau', 'none'
        optimizer='adam',
        learning_rate=0.001,
        decay_factor=None,
        decay_steps=None,
        start_decay_step=None,
        time_major=False,
        infer_helper='sample',  # 'greedy', 'sample'
        max_gradient_norm=5.0,
        src_max_len=85,
        tgt_max_len=85,
        num_epochs=8,
        src_filename=DATA_PATH + 'train2.vi',  # '(..).2.de',
        tgt_filename=DATA_PATH + 'train2.en',  # '(..).2.en',
        test_src_filename=DATA_PATH + 'tst20132.vi',
        test_tgt_filename=DATA_PATH + 'tst20132.en',
        src_vocab_filename=DATA_PATH + 'vocab2.vi',
        tgt_vocab_filename=DATA_PATH + 'vocab2.en',
        checkpoints_path='./checkpoints/',
        eval_filename='/tmp/translations',
        src_eos_id=2,
        tgt_sos_id=1,
        tgt_eos_id=2,
        save_memory=False)
    hparams.update(kwargs)
    return hparams


def update_hparams(hparams, **kwargs):
    hparams.update(kwargs)


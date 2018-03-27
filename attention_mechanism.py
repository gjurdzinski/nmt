from __future__ import print_function
from __future__ import division
import tensorflow as tf
import utils
import dataset_utils
import hyper_params
import model_wrapper


def main():
    hparams_to_test = [
        {
            'bi_reduce': 'layers',
            'infer_helper': 'greedy',
            'attention': 'none',
            'optimizer': 'adam',
            'num_epochs': 7,
            'bidirectional_encoder': False,
            'checkpoints_path': './check_no_attention/'},
        {
            'bi_reduce': 'layers',
            'infer_helper': 'greedy',
            'attention': 'luong',
            'optimizer': 'gd',
            'learning_rate': 1.0,
            'decay_factor': 0.8,
            'start_decay_step': 15000,
            'decay_steps': 2000,
            'num_epochs': 12,
            'checkpoints_path': './check_attention'},
    ]

    for to_test in hparams_to_test:
        hparams = hyper_params.get_hparams(**to_test)

        nmt_model = model_wrapper.NMTModel(hparams)
        try:
            nmt_model.train(num_epochs=hparams.num_epochs, eval_every=500)
        except KeyboardInterrupt:
            utils.print_time('Interrupted by user')
        except Exception as e:
            utils.print_time('Other exception: %s' % str(e))
        except:
            utils.print_time('Other exception')


if __name__ == '__main__':
    main()

# Copied from tensorflow tutorial file:
# https://github.com/tensorflow/nmt/blob/tf-1.4/nmt/utils/evaluation_utils.py
import codecs
import os
import re
import subprocess

import tensorflow as tf
import bleu as _bleu


def _clean(sentence):
    """Clean and handle BPE outputs."""
    sentence = sentence.strip()
    sentence = re.sub("@@ ", "", sentence)
    return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def bleu(ref_file, trans_file):
    """Compute BLEU scores and handling BPE."""
    max_order = 4
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(tf.gfile.GFile(reference_filename,
                                                      "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
        for line in fh:
            line = _clean(line)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = _bleu.compute_bleu(
          per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score

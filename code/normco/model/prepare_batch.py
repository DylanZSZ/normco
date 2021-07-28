import numpy as np
from .reader_utils import text_to_batch
from ..utils.text_processing import tokens_to_ids

# def load_text_batch(examples, vocab, id_dict, maxlen=20, precleaned=False):
#     words = []
#     lens = []
#     disease_ids = []
#
#     for ex in examples:
#         w_curr = []
#
#         words_curr, word_len = text_to_batch(ex[0],
#                                              vocab,
#                                              maxlen=maxlen,
#                                              precleaned=precleaned)
#         lens.append(word_len)
#         words.append(np.asarray(words_curr))
#
#         disease_ids.append(np.asarray([id_dict[i] for i in ex[1:]]))
#
#     seq_len = len(words)
#
#     return np.asarray(words), np.asarray(lens), np.asarray(disease_ids), seq_len

def load_text_batch(examples, vocab, maxlen=20, precleaned=False):
    words = []
    lens = []
    ids = []

    for ex in examples:
        if ex[1] == []:

            pass
        else:
            w_curr = []
            words_curr, word_len = text_to_batch(ex[0],
                                                 vocab,
                                                 maxlen=maxlen,
                                                 precleaned=precleaned)
            lens.append(np.asarray([word_len]))
            words.append(np.asarray([words_curr]))
            ids.append(np.asarray([ex[1]]))
    seq_len = len(words)

    return np.asarray(words), np.asarray(lens), np.asarray(ids), seq_len
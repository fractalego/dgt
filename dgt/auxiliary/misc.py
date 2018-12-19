import os
import pickle
import logging

from dgt.metric import GloveMetric
from gensim.models import KeyedVectors


def get_metric_or_save_pickle(path, filename, pickle_name):
    if not os.path.isfile(os.path.join(path, pickle_name)):
        logging.info('Creating pickled metric from Glove')
        metric = GloveMetric(KeyedVectors.load_word2vec_format(os.path.join(path, filename)))
        pickle.dump(metric, open(os.path.join(path, pickle_name), 'wb'))

    else:
        logging.info('Using pickled metric')
        metric = pickle.load(open(os.path.join(path, pickle_name), 'rb'))

    return metric

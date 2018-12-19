import os
import pickle
import json

from dgt import DGT, set_global_device

_path = os.path.dirname(__file__)
_gradient_test_filename = os.path.join(_path, '../data/two_gradient_rules_test.json')

from dgt.metric import GloveMetric
from gensim.models import KeyedVectors

#_word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/glove.txt'))
#_metric = GloveMetric(_word2vec_model)
#pickle.dump(_metric, open(os.path.join(_path, '../data/metric.pickle'), 'wb'))
_metric = pickle.load(open(os.path.join(_path, '../data/metric.pickle'), 'rb'))

set_global_device('cpu')

if __name__ == '__main__':
    dgt = DGT(_metric)
    dgt.fit(json.load(open(_gradient_test_filename)))
    print(dgt.get_all_rules_with_weights())
    dgt.save(open(os.path.join(_path, '../data/saved.json'), 'w'))

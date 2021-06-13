import os
import json

from dgt import DGT, set_global_device
from dgt.auxiliary.misc import get_metric_or_save_pickle

_path = os.path.dirname(__file__)
#_gradient_test_filename = os.path.join(_path, '../data/politics.json')
_gradient_test_filename = os.path.join(_path, '../data/more_gradient_rules.json')

_metric = get_metric_or_save_pickle(_path, '../data/glove.txt', '../data/metric.pickle')

set_global_device('cpu')

if __name__ == '__main__':
    dgt = DGT(_metric)
    dgt.fit(json.load(open(_gradient_test_filename)))
    print(dgt.get_all_rules_with_weights())
    dgt.save(open(os.path.join(_path, '../data/saved.json'), 'w'))

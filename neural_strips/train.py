import os
import pickle
import json

from neural_strips.inference.forward_inference import ForwardInference
from neural_strips.utils import train_all_paths
from neural_strips.utils import get_data_goal_knowledge_from_json
from neural_strips.utils import print_all_the_paths
from neural_strips.utils import print_predicates
from neural_strips.utils import print_all_the_rules_with_weights

_path = os.path.dirname(__file__)
_gradient_test_filename = os.path.join(_path, '../data/two_gradient_rules_test.json')

# from neural_strips.metric import GloveMetric
# _metric = GloveMetric()
# pickle.dump(_metric, open(os.path.join(_path, '../data/metric.pickle'), 'wb'))
_metric = pickle.load(open(os.path.join(_path, '../data/metric.pickle'), 'rb'))

if __name__ == '__main__':
    data, goals, k = get_data_goal_knowledge_from_json(json.load(open(_gradient_test_filename)), _metric)

    for fact, goal in zip(data, goals):
        print_predicates(k)

        fw = ForwardInference(data=fact, knowledge=k)
        end_drs = fw.compute()

        print_all_the_paths(end_drs)
        train_all_paths(_metric, k, end_drs, goal)

        print_all_the_rules_with_weights(k)

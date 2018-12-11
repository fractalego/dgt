import os
import pickle
import json
import unittest

from neural_strips.drt.drs import Drs
from neural_strips.inference.forward_inference import ForwardInference
from neural_strips.utils import train_all_paths, get_string_with_all_the_rules_with_weights
from neural_strips.utils import get_data_goal_knowledge_from_json

_path = os.path.dirname(__file__)
_two_gradient_rules_test_filename = os.path.join(_path, '../data/two_gradient_rules_test.json')

# from neural_strips.metric import GloveMetric
# _metric = GloveMetric()
# pickle.dump(_metric, open(os.path.join(_path, '../data/metric.pickle'), 'wb'))
_metric = pickle.load(open(os.path.join(_path, '../data/metric.pickle'), 'rb'))




class Tests(unittest.TestCase):
    def two_rules_test(self):
        data, goals, k = get_data_goal_knowledge_from_json(json.load(open(_two_gradient_rules_test_filename)), _metric)

        for fact, goal in zip(data, goals):
            fw = ForwardInference(data=fact, knowledge=k)
            end_drs = fw.compute()

            train_all_paths(_metric, k, end_drs, goal)

            expected_str_list = ["MATCH apple(a), fruit(b), is(a,b) CREATE apple(a), animal(d2), not(a,d2) DELETE b, a-b RETURN __RESULT__",
                                 "MATCH apple(a), animal(d), not(a,d) CREATE entity(a), delicious(d3), is(a,d3) DELETE d, a-d RETURN __RESULT__  *GRADIENT RULE*",
                                 "MATCH apple(a), animal(d), not(a,d) CREATE apple(a2), animal(d3), not(a2,d3) DELETE a, d, a-d RETURN __RESULT__  *GRADIENT RULE*"]
            result_str_list = get_string_with_all_the_rules_with_weights(k)
            self.assertEqual(expected_str_list[0], result_str_list[0])

    def threshold_can_be_written_in_rules(self):
        drs = Drs.create_from_predicates_string('MATCH apple>0.5(a), is(a,b), fruit(b) CREATE apple(a), not(a,d2), animal(d2) DELETE b, a-b')
        string = drs.predicates()
        expected_str = 'MATCH apple>0.5(a), fruit>0.6(b), is(a,b) CREATE apple(a), animal(d2), not(a,d2) DELETE b, a-b RETURN __RESULT__  tensor([1.])'
        self.assertEqual(expected_str, string)

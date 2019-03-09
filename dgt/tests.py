import os
import json
import unittest

from dgt import DGT
from dgt.auxiliary.misc import get_metric_or_save_pickle
from dgt.graph.graph import Graph
from dgt.utils import get_relations_embeddings_dict_from_json

_path = os.path.dirname(__file__)
_two_gradient_rules_test_filename = os.path.join(_path, '../data/two_gradient_rules_test.json')
_two_gradient_rules_with_sharp_test_filename = os.path.join(_path, '../data/two_gradient_rules_test_with_sharp.json')

_metric = get_metric_or_save_pickle(_path, '../data/glove.txt', '../data/metric.pickle')


class Tests(unittest.TestCase):
    def test_two_rules(self):
        dgt = DGT(_metric)
        dgt.fit(json.load(open(_two_gradient_rules_test_filename)), epochs=200)
        result_str_list = dgt.get_all_rules_with_weights()
        expected_str_list = [
            "MATCH apple(a), fruit(b), is(a,b) CREATE apple(a), animal(d2), not(a,d2) DELETE b, a-b RETURN __RESULT__",
            "MATCH apple(a), fruit(d), is(a,d) CREATE entity(a), delicious(d3), is(a,d3) DELETE d, a-d RETURN __RESULT__",
            "MATCH apple(a), animal(d), is(a,d) CREATE apple(a2), delicious(d3), is(a2,d3) DELETE a, d, a-d RETURN __RESULT__"]
        print(expected_str_list[2])
        print(result_str_list[2])
        self.assertEqual(expected_str_list[0], result_str_list[0])
        self.assertEqual(expected_str_list[1], result_str_list[1])
        self.assertEqual(expected_str_list[2], result_str_list[2])

    def test_two_rules_with_sharp(self):
        dgt = DGT(_metric)
        dgt.fit(json.load(open(_two_gradient_rules_with_sharp_test_filename)), epochs=100)
        result_str_list = dgt.get_all_rules_with_weights()
        expected_str_list = [
            "MATCH apple(a), fruit(b), is(a,b) CREATE apple(a), animal(d2), not(a,d2) DELETE b, a-b RETURN __RESULT__",
            "MATCH apple(a), nutritious(d), not(a,d) CREATE entity(a), delicious(d3), is(a,d3) DELETE d, a-d RETURN __RESULT__",
            "MATCH apple(a), fruit(d), not(a,d) CREATE apple(a2), nutritious(d3), not(a2,d3) DELETE a, d, a-d RETURN __RESULT__"]
        self.assertEqual(expected_str_list[0], result_str_list[0])
        self.assertEqual(expected_str_list[1], result_str_list[1])
        self.assertEqual(expected_str_list[2], result_str_list[2])

    def test_threshold_can_be_written_in_graph(self):
        relations_metric = get_relations_embeddings_dict_from_json(json.load(open(_two_gradient_rules_test_filename)))
        drs = Graph.create_from_predicates_string('apple>0.5(a), is(a,b), fruit>0.6(b)', _metric, relations_metric,
                                                  gradient=True)
        string = drs.predicates(print_threshold=True)
        expected_str = 'apple>0.5(a), fruit>0.6(b), is>0.9(a,b)'
        self.assertEqual(expected_str, string)

    def test_relations_are_embedded(self):
        relations_metric = get_relations_embeddings_dict_from_json(json.load(open(_two_gradient_rules_test_filename)))
        self.assertEqual(relations_metric.get_vector_index('is'), 0)
        self.assertEqual(relations_metric.get_vector_index('not'), 1)

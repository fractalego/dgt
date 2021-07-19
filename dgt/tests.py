import os
import json
import unittest

from dgt import DGT
from dgt.auxiliary.misc import get_metric_or_save_pickle
from dgt.graph.graph import Graph
from dgt.utils import get_relations_embeddings_dict_from_json

_path = os.path.dirname(__file__)
_two_gradient_rules_test_filename = os.path.join(_path, '../data/more_gradient_rules.json')
_politics_filename = os.path.join(_path, '../data/politics.json')

_metric = get_metric_or_save_pickle(_path, '../data/glove.txt', '../data/metric.pickle')


class Tests(unittest.TestCase):
    def test_two_rules(self):
        dgt = DGT(_metric)
        dgt.fit(json.load(open(_two_gradient_rules_test_filename)), epochs=20)
        result_str_list = dgt.get_all_rules_with_weights(print_threshold=False)
        expected_str_list = [
            "MATCH fruit(a), round(b), delicious(c), is(a,b), is(a,c) CREATE entity(b), entity(c), and(b,c) DELETE a-b, a-c RETURN __RESULT__",
            "MATCH round(a), delicious(b), and(a,b) CREATE fruit(c), apple(d), is(c,d) DELETE a, b, a-b RETURN __RESULT__"]
        print(expected_str_list[0])
        print(result_str_list[0])
        self.assertEqual(expected_str_list[0], result_str_list[0])
        self.assertEqual(expected_str_list[1], result_str_list[1])

    def test_single_rule(self):
        dgt = DGT(_metric)
        dgt.fit(json.load(open(_politics_filename)), epochs=20)
        result_str_list = dgt.get_all_rules_with_weights(print_threshold=False)
        expected_str_list = [
            "MATCH person(a), person(b), first-lady(c), spouse(a,b), is-a(a,c) CREATE entity(b), president(d), profession(b,d) RETURN __RESULT__"]
        print(expected_str_list[0])
        print(result_str_list[0])
        self.assertEqual(expected_str_list[0], result_str_list[0])

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
        self.assertEqual(relations_metric.get_vector_index('and'), 1)

import os
import pickle
import json
import unittest

from gensim.models import KeyedVectors

from dgt.graph.graph import Graph
from dgt.inference.forward_inference import ForwardInference
from dgt.utils import train_all_paths, get_string_with_all_the_rules_with_weights, \
    get_relations_embeddings_dict_from_json
from dgt.utils import get_data_goal_knowledge_from_json

_path = os.path.dirname(__file__)
_two_gradient_rules_test_filename = os.path.join(_path, '../data/two_gradient_rules_test.json')
_two_gradient_rules_with_sharp_test_filename = os.path.join(_path, '../data/two_gradient_rules_test_with_sharp.json')

# from dgt.metric import GloveMetric
# _word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/glove.txt'))
# _metric = GloveMetric(_word2vec_model)
# pickle.dump(_metric, open(os.path.join(_path, '../data/metric.pickle'), 'wb'))
_metric = pickle.load(open(os.path.join(_path, '../data/metric.pickle'), 'rb'))


class Tests(unittest.TestCase):
    def test_two_rules(self):
        relations_metric = get_relations_embeddings_dict_from_json(json.load(open(_two_gradient_rules_test_filename)))
        data, goals, k = get_data_goal_knowledge_from_json(json.load(open(_two_gradient_rules_test_filename)), _metric,
                                                           relations_metric)

        for fact, goal in zip(data, goals):
            fw = ForwardInference(data=fact, knowledge=k)
            end_graph = fw.compute()

            train_all_paths(_metric, relations_metric, k, end_graph, goal, epochs=100, step=5e-3)

            expected_str_list = [
                "MATCH apple(a), fruit(b), is(a,b) CREATE apple(a), animal(d2), not(a,d2) DELETE b, a-b RETURN __RESULT__",
                "MATCH apple(a), fruit(d), is(a,d) CREATE entity(a), delicious(d3), is(a,d3) DELETE d, a-d RETURN __RESULT__  *GRADIENT RULE*",
                "MATCH apple(a), animal(d), is(a,d) CREATE apple(a2), delicious(d3), is(a2,d3) DELETE a, d, a-d RETURN __RESULT__  *GRADIENT RULE*"]
            result_str_list = get_string_with_all_the_rules_with_weights(k)
            self.assertEqual(expected_str_list[0], result_str_list[0])
            self.assertEqual(expected_str_list[1], result_str_list[1])
            self.assertEqual(expected_str_list[2], result_str_list[2])

    def test_two_rules_with_sharp(self):
        relations_metric = get_relations_embeddings_dict_from_json(
            json.load(open(_two_gradient_rules_with_sharp_test_filename)))
        data, goals, k = get_data_goal_knowledge_from_json(json.load(open(_two_gradient_rules_with_sharp_test_filename)), _metric,
                                                           relations_metric)

        for fact, goal in zip(data, goals):
            fw = ForwardInference(data=fact, knowledge=k)
            end_graph = fw.compute()

            train_all_paths(_metric, relations_metric, k, end_graph, goal, epochs=100, step=5e-3)

            expected_str_list = [
                "MATCH apple(a), fruit(b), is(a,b) CREATE apple(a), animal(d2), not(a,d2) DELETE b, a-b RETURN __RESULT__",
                "MATCH apple(a), nutritious(d), not(a,d) CREATE entity(a), delicious(d3), is(a,d3) DELETE d, a-d RETURN __RESULT__  *GRADIENT RULE*",
                "MATCH apple(a), fruit(d), not(a,d) CREATE apple(a2), nutritious(d3), not(a2,d3) DELETE a, d, a-d RETURN __RESULT__  *GRADIENT RULE*"]

            result_str_list = get_string_with_all_the_rules_with_weights(k)

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
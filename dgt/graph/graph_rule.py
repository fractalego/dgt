import torch

import numpy as np

from igraph import Graph as iGraph
from parvusdb import GraphDatabase
from parvusdb.utils.code_container import DummyCodeContainerFactory

from dgt.graph.graph import Graph, create_graph_from_predicates
from dgt.auxiliary.config import device
from .node_matcher import VectorNodeMatcher


def convert_special_characters_to_spaces(line):
    line = line.replace('\t', ' ')
    line = line.replace('\n', ' ')
    return line


class GraphRule:
    def __init__(self, text, metric, relations_metric, gradient=True):
        self.action_list = ['MATCH ', 'CREATE ', 'DELETE ', 'RETURN', 'SET ', 'WHERE ']
        self.metric = metric
        self.relations_metric = relations_metric
        self.gradient = gradient
        self.action_pairs = self.__pre_process_text(text, gradient=gradient)
        if '__RESULT__' not in text:
            self.action_pairs.append(('RETURN', '__RESULT__'))
        vector = torch.Tensor(np.array([1.])).to(device)
        self.weight = torch.autograd.Variable(vector, requires_grad=gradient)
        self.max_repeat = 5

    def test(self):
        g = iGraph(directed=True)
        self.apply(g)

    def apply(self, g):
        if not isinstance(g, iGraph):
            raise TypeError("GraphRule.apply_to_graph() needs an igraph.Graph as an argument")
        db = GraphDatabase(g,
                           node_matcher=VectorNodeMatcher(self.metric, self.relations_metric, self.gradient),
                           code_container_factory=DummyCodeContainerFactory())
        text = self.__create_text(self.action_pairs) + ';'
        lst = db.query(str(text), repeat_n_times=self.max_repeat)
        if lst and lst[0]['__RESULT__']:
            return True
        return False

    def get_hypothesis(self):
        for item in self.action_pairs:
            if item[0] == 'MATCH':
                return item[1]

    def get_consequence(self):
        for item in self.action_pairs:
            if item[0] == 'CREATE':
                return item[1]

    def __str__(self):
        return self.__create_text(self.action_pairs)

    def predicates(self, print_threshold=True, print_gradient=False):
        return self.__create_text_with_predicates(self.action_pairs, print_threshold, print_gradient)

    def __get_action_graph_pairs_from_query(self, query):
        import re

        query = convert_special_characters_to_spaces(query)
        graph_list = re.split('|'.join(self.action_list), query)
        query_list_positions = [query.find(graph) for graph in graph_list]
        query_list_positions = query_list_positions
        query_list_positions = query_list_positions
        action_list = [query[query_list_positions[i] + len(graph_list[i]):query_list_positions[i + 1]].strip()
                       for i in range(len(graph_list) - 1)]
        graph_list = graph_list[1:]
        return [(item[0], item[1]) for item in zip(action_list, graph_list)]

    def __pre_process_text(self, text, gradient):
        action_pairs = self.__get_action_graph_pairs_from_query(text)
        new_action_pairs = []
        for action, context in action_pairs:
            if action in ['MATCH', 'CREATE']:
                g = create_graph_from_predicates(context)
                new_action_pairs.append((action, Graph(g, self.metric, self.relations_metric, gradient=gradient)))
            else:
                new_action_pairs.append((action, context))
        return new_action_pairs

    def __create_text(self, action_pairs):
        string = ''
        for action, context in action_pairs:
            string += action + ' '
            string += str(context) + ' '
        return string

    def __create_text_with_predicates(self, action_pairs, print_threshold, print_gradient):
        string = ''
        for action, context in action_pairs:
            string += action + ' '
            if type(context) is Graph:
                do_print_thresholds = False
                if action == 'MATCH' and print_threshold:
                    do_print_thresholds = True
                string += context.predicates(do_print_thresholds) + ' '
            else:
                string += context + ' '
        if print_gradient and self.gradient:
            string += ' ' + '*GRADIENT RULE*'
        return string

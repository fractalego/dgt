import logging

from igraph import Graph as iGraph
from parvusdb import GraphDatabase
from parvusdb.utils.code_container import DummyCodeContainer
from parvusdb.utils.match import Match, MatchException

from .node_matcher import VectorNodeMatcher

_logger = logging.getLogger()


class GraphMatcher:
    def __init__(self, small_graph, metric, relations_metric):
        self.small_graph = small_graph
        self.metric = metric
        self.relations_metric = relations_metric

    def apply(self, g):
        if not isinstance(g, iGraph):
            raise TypeError("GraphRule.apply_to_graph() needs an igraph.Graph as an argument")
        db = GraphDatabase(g, node_matcher=VectorNodeMatcher(self.metric, self.relations_metric))
        rule = 'MATCH ' + str(self.small_graph) + ' RETURN __RESULT__;'
        lst = db.query(rule)
        if lst and lst[0]:
            return True
        return False


class GraphWeightedMatch:
    def __init__(self, big_graph, metric, relations_metric):
        self.big_graph = big_graph
        self.metric = metric
        self.relations_metric = relations_metric

    def apply(self, g):
        if not isinstance(g, iGraph):
            raise TypeError("GraphRule.apply_to_graph() needs an igraph.Graph as an argument")
        match = Match(matching_code_container=DummyCodeContainer(),
                      node_matcher=VectorNodeMatcher(self.metric, self.relations_metric, gradient=False))
        big_graph = self.big_graph._g
        try:
            matching_variables = match.get_variables_substitution_dictionaries(g, big_graph)
            w = 0
            for k, v in matching_variables[0].items():
                rindex = big_graph.vs['name' == v]['vector']
                lindex = g.vs['name' == k]['vector']
                w += self.metric.indices_dot_product(lindex, rindex)
            return w
        except MatchException as e:
            _logger.warning('Cannot find matching variables %s', str(e))
        return 0

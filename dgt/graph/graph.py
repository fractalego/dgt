from igraph import Graph as iGraph
from parvusdb import convert_graph_to_string


def convert_graph_to_string_with_predicates(g, metric, relations_metric, print_thresholds):
    drt_string = ""
    for vertex in g.vs:
        attributes = vertex.attributes()
        attributes.pop('name')
        drt_string += metric.get_most_similar_string_from_vector(
            metric.get_vector_from_index(attributes['vector']).cpu().detach().numpy())
        if print_thresholds:
            drt_string += '>' + str(metric.get_threshold_from_index(attributes['vector']).cpu().detach().numpy()[0])
        drt_string += '('
        drt_string += vertex['name']
        drt_string += ')'
        drt_string += ', '
    for edge in g.es:
        attributes = edge.attributes()
        attributes.pop('name')
        drt_string += relations_metric.get_most_similar_string_from_vector(
            relations_metric.get_vector_from_index(attributes['rvector']).cpu().detach().numpy())
        if print_thresholds:
            drt_string += '>' + str(relations_metric.get_threshold_from_index(attributes['rvector']).cpu().detach().numpy()[0])
        drt_string += '('
        drt_string += g.vs[edge.tuple[0]]['name']
        drt_string += ','
        drt_string += g.vs[edge.tuple[1]]['name']
        drt_string += ')'
        drt_string += ', '
    drt_string = drt_string[:-2]

    return drt_string


def is_edge(string):
    return string.find(',') != -1


def create_graph_from_predicates(graph_string, directed=True):
    g = iGraph(directed=directed)
    line = graph_string.replace(' ', '')
    line = line.replace('\n', '')
    predicates_strings = line.split("),")
    vertices_to_add = []
    edges_to_add = []
    for predicate in predicates_strings:
        predicate = predicate.replace(')', '')
        attributes_str, name_str = predicate.split("(")
        attributes_dict = {}
        if not is_edge(name_str):
            attributes_dict['name'] = name_str
            attributes_dict['text'] = attributes_str
            vertices_to_add.append(attributes_dict)
        else:
            attributes_dict['name'] = name_str.replace(',', '-')  # the name of a relation is a,b -> a-b
            attributes_dict['relation'] = attributes_str
            source, target = name_str.split(',')
            edges_to_add.append((source, target, attributes_dict))
    for attributes_dict in vertices_to_add:
        g.add_vertex(**attributes_dict)
    for source, target, attributes_dict in edges_to_add:
        g.add_edge(source, target, **attributes_dict)
    return g


class Graph:
    def __init__(self, g, metric, relations_metric, gradient=False):
        if not isinstance(g, iGraph):
            raise TypeError("Graph needs an igraph.Graph as an argument")
        self._metric = metric
        self._relations_metric = relations_metric
        self.gradient = gradient
        for vertex in g.vs:
            text = vertex['text']
            try:
                if not vertex['vector']:
                    vertex['vector'] = self._metric.get_vector_index(text, gradient)
            except:
                vertex['vector'] = self._metric.get_vector_index(text, gradient)
        for relation in g.es:
            text = relation['relation']
            try:
                if not relation['rvector']:
                    relation['rvector'] = self._relations_metric.get_vector_index(text, gradient)
            except:
                relation['rvector'] = self._relations_metric.get_vector_index(text, gradient)
        self._g = g

    def __str__(self):
        return convert_graph_to_string(self._g)

    def predicates(self, print_threshold=False):
        return convert_graph_to_string_with_predicates(self._g, self._metric, self._relations_metric, print_threshold)

    @staticmethod
    def create_from_predicates_string(string, metric, relations_metric, gradient):
        return Graph(create_graph_from_predicates(string), metric, relations_metric, gradient)

    @staticmethod
    def create_empty(metric, relations_metric):
        return Graph(iGraph(directed=True), metric, relations_metric)

    def copy(self):
        return Graph(self._g.as_directed(), self._metric, self._relations_metric)

    def visit(self, function):
        return function.apply(self._g)

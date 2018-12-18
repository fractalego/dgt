from igraph import Graph


class GraphNERCleaner:
    def __init__(self, words_without_entity):
        self._words_without_entity = words_without_entity

    def apply(self, g):
        if not isinstance(g, Graph):
            raise TypeError("GraphRule.apply_to_graph() needs an igraph.Graph as an argument")
        for vertex in g.vs:
            if vertex['word'] in self._words_without_entity:
                vertex['entity'] = ''
        return Graph(g)

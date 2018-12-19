import json

from dgt.inference import ForwardInference
from dgt.utils import get_relations_embeddings_dict_from_json, get_data_goal_knowledge_from_json, train_all_paths, \
    print_predicates, get_string_with_all_the_rules_with_weights


class DGT:
    def __init__(self, glove_metric):
        self._metric = glove_metric
        self._relations_metric = None
        self._data = None
        self._goals = None
        self._k = None

    def fit(self, json_dict, epochs=50, step=5e-3):
        self.__load_from_json(json_dict)
        for fact, goal in zip(self._data, self._goals):
            fw = ForwardInference(data=fact, knowledge=self._k)
            end_graphs = fw.compute()
            train_all_paths(self._metric, self._relations_metric, self._k, end_graphs, goal, epochs, step)

    def predict(self, fact):
        fw = ForwardInference(data=fact, knowledge=self._k)
        end_graphs = fw.compute()
        return [{'graph': item[0], 'score': item[1]} for item in end_graphs]

    def predict_best(self, fact):
        fw = ForwardInference(data=fact, knowledge=self._k)
        end_graphs = fw.compute()
        if not end_graphs:
            return None
        end_graphs = end_graphs[0]
        return [{'graph': item[0], 'score': item[1]} for item in end_graphs]

    def save(self, filestream):
        to_return = {'facts': [item.predicates(print_threshold=False) for item in self._data],
                     'goals': [item.predicates(print_threshold=False) for item in self._goals],
                     'relations': [word for word in self._relations_metric._model.index2word],
                     'non_trainable_rules': [rule[0].predicates() for rule in self._k.get_all_rules()]
                     }
        json.dump(to_return, filestream, indent=2)

    def print_all_rules(self):
        print_predicates(self._k)

    def get_all_rules_with_weights(self, print_gradient=False):
        return get_string_with_all_the_rules_with_weights(self._k, print_gradient=False)

    def __load_from_json(self, json_dict):
        self._relations_metric = get_relations_embeddings_dict_from_json(json_dict)
        self._data, self._goals, self._k = get_data_goal_knowledge_from_json(json_dict,
                                                                             self._metric,
                                                                             self._relations_metric)

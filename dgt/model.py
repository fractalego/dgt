import json
import random

from dgt.inference import ForwardInference
from dgt.utils import get_relations_embeddings_dict_from_json, get_data_goal_knowledge_from_json, train_all_paths, \
    print_predicates, get_string_with_all_the_rules_with_weights


class DGT:
    _clamp_threshold = 0.7
    _max_depth = 2

    def __init__(self, glove_metric):
        self._metric = glove_metric

    @property
    def goals(self):
        return self._goals

    def fit(self, json_dict, epochs=20, step=5e-3):
        self.__load_from_json(json_dict)
        shifts_and_finished_paths = []
        for fact, goal in zip(self._data, self._goals):
            for i in range(100):
                permutations = [int(random.uniform(0, 10)) for _ in range(self._max_depth)]
                permutations.append(0)
                print('Permutation number', i, '=>', permutations)
                fw = ForwardInference(data=fact, knowledge=self._k, permutation_shift=permutations,
                                      max_depth=self._max_depth)
                end_graphs = fw.compute()
                finished_paths = train_all_paths(self._metric, self._relations_metric, self._k, end_graphs, goal,
                                                 permutations,
                                                 self._clamp_threshold, epochs, step)
                if finished_paths:
                    train_all_paths(self._metric, self._relations_metric, self._k, finished_paths, goal,
                                    permutations,
                                    self._clamp_threshold, 5, step)
                    shifts_and_finished_paths.append((permutations, finished_paths, goal))
                    break

        for _ in range(20):
            random.shuffle(shifts_and_finished_paths)
            for permutations, path, goal in shifts_and_finished_paths:
                train_all_paths(self._metric, self._relations_metric, self._k, path, goal,
                                permutations,
                                self._clamp_threshold, 1, 5e-3)

    def predict(self, fact):
        fw = ForwardInference(data=fact, knowledge=self._k, permutation_shift=0)
        end_graphs = fw.compute()
        return [{'graph': item[0], 'score': item[1]} for item in end_graphs]

    def predict_best(self):
        to_return = []
        for fact in self._data:
            fw = ForwardInference(data=fact, knowledge=self._k, permutation_shift=0)
            end_graphs = fw.compute()
            if end_graphs:
                graph = end_graphs[0]
                to_return.append({'graph': graph[0], 'score': graph[1]})
            else:
                to_return.append(None)
        return to_return

    def save(self, filestream):
        to_return = {'facts': [item.predicates(print_threshold=False) for item in self._data],
                     'goals': [item.predicates(print_threshold=False) for item in self._goals],
                     'relations': [word for word in self._relations_metric._model.index2word],
                     'non_trainable_rules': [rule[0].predicates() for rule in self._k.get_all_rules()]
                     }
        json.dump(to_return, filestream, indent=2)

    def print_all_rules(self):
        print_predicates(self._k)

    def get_all_rules_with_weights(self, print_threshold=True, print_gradient=False):
        return get_string_with_all_the_rules_with_weights(self._k,
                                                          print_threshold=print_threshold,
                                                          print_gradient=print_gradient)

    def __load_from_json(self, json_dict):
        self._relations_metric = get_relations_embeddings_dict_from_json(json_dict)
        self._data, self._goals, self._k = get_data_goal_knowledge_from_json(json_dict,
                                                                             self._metric,
                                                                             self._relations_metric)

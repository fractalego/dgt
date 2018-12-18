import os
import pickle
import json

from time import time
from gensim.models import KeyedVectors

from neural_strips.inference.forward_inference import ForwardInference
from neural_strips.utils import train_all_paths, get_relations_embeddings_dict_from_json
from neural_strips.utils import get_data_goal_knowledge_from_json
from neural_strips.utils import print_all_the_paths
from neural_strips.utils import print_predicates
from neural_strips.utils import print_all_the_rules_with_weights

_path = os.path.dirname(__file__)
_gradient_test_filename = os.path.join(_path, '../data/two_gradient_rules_test_with_sharp.json')

# from neural_strips.metric import GloveMetric
# _word2vec_model = KeyedVectors.load_word2vec_format(os.path.join(_path, '../data/glove.txt'))
# _metric = GloveMetric(_word2vec_model)
# pickle.dump(_metric, open(os.path.join(_path, '../data/metric.pickle'), 'wb'))
_metric = pickle.load(open(os.path.join(_path, '../data/metric.pickle'), 'rb'))

if __name__ == '__main__':
    relations_metric = get_relations_embeddings_dict_from_json(json.load(open(_gradient_test_filename)))
    data, goals, k = get_data_goal_knowledge_from_json(json.load(open(_gradient_test_filename)), _metric,
                                                       relations_metric)

    for fact, goal in zip(data, goals):
        #print_predicates(k)

        fw = ForwardInference(data=fact, knowledge=k)
        end_graph = fw.compute()

        #print_all_the_paths(end_graph)
        start = time()
        train_all_paths(_metric, relations_metric, k, end_graph, goal, epochs=50, step=5e-3)
        print('Total time:', time() - start)

        print_all_the_rules_with_weights(k)

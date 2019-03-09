import copy
import os
import torch
import time
import numpy as np

from dgt.auxiliary.config import device

torch.manual_seed(time.time())


class MetricBase(object):
    _substitution_dict = {}

    def similarity(self, lhs, rhs):
        lvect = self._get_vector(lhs)
        rvect = self._get_vector(rhs)
        if lhs not in self._substitution_dict:
            return np.dot(lvect, rvect) / np.linalg.norm(lvect) / np.linalg.norm(rvect)
        lst = self._substitution_dict[lhs]
        distance_lst = []
        for item in lst:
            lvect = self._get_vector(item)
            distance_lst.append(np.dot(lvect, rvect) / np.linalg.norm(lvect) / np.linalg.norm(rvect))
        return min(distance_lst)

    def add_substitution(self, name, lst):
        self._substitution_dict[name] = lst

    def get_vector_index(self, word):
        return 0

    def _get_vector(self, word):
        return np.array([0.] * 300)


class GloveMetric(MetricBase):
    _last_index = -1

    def __init__(self, word2vec_model, threshold=0.6):
        self._vector_map = {}
        self._path = os.path.dirname(__file__)
        self._model = word2vec_model
        self._vector_matching_threshold = threshold
        self.vector_size = word2vec_model.vector_size

    def get_vector_index(self, text, gradient=True):
        self._last_index += 1
        threshold = self._vector_matching_threshold
        word = text
        if '>' in text:
            word, threshold = text.split('>')
            threshold = float(threshold)

        if word and word[0] == '#':
            word = word[1:]
            gradient = False

        threshold = torch.autograd.Variable(torch.Tensor(np.array([threshold])).to(device),
                                            requires_grad=gradient)
        if word == '*':
            vector = torch.randn(self.vector_size).to(device)
        else:
            vector = torch.autograd.Variable(torch.Tensor(self._get_vector(word)).to(device), requires_grad=gradient)

        embedding = copy.deepcopy(torch.autograd.Variable(vector / vector.norm(2), requires_grad=gradient))
        self._vector_map[self._last_index] = (embedding, threshold)
        return self._last_index

    def copy_vector_index(self, index, gradient):
        old_index, threshold = self._vector_map[index]
        self._last_index += 1
        embedding = torch.autograd.Variable(self._vector_map[old_index], requires_grad=gradient)
        self._vector_map[self._last_index] = (embedding, threshold)
        return self._last_index

    def get_vector_from_index(self, index):
        return self._vector_map[index][0]

    def get_threshold_from_index(self, index):
        return self._vector_map[index][1]

    def indices_have_similar_vectors(self, lindex, rindex):
        _, rthreshold = self._vector_map[rindex]
        #print(self.get_most_similar_string_from_vector(
        #    self.get_vector_from_index(lindex).cpu().detach().numpy()))
        #print(self.get_most_similar_string_from_vector(
        #    self.get_vector_from_index(rindex).cpu().detach().numpy()))
        return self.indices_dot_product(lindex, rindex) > rthreshold

    def indices_dot_product(self, lindex, rindex):
        import torch
        lvect = self._vector_map[lindex][0]
        rvect, _ = self._vector_map[rindex]
        return torch.dot(lvect, rvect) / lvect.norm(2) / rvect.norm(2)

    def get_indexed_vectors(self):
        return [item[0] for item in list(self._vector_map.values())]

    def get_indexed_thresholds(self):
        return [item[1] for item in list(self._vector_map.values())]

    def get_most_similar_string_from_vector(self, vector):
        return self._model.most_similar(positive=[vector])[0][0]

    def copy(self):
        import copy
        to_return = GloveMetric(self._model, self._vector_matching_threshold)
        to_return._vector_map = copy.deepcopy(self._vector_map)
        return to_return

    # Private

    def _get_vector(self, word):
        try:
            return self._model[word]
        except:
            return self._model['entity']

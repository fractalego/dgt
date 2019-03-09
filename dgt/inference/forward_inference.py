import logging

from dgt.graph.graph_matcher import GraphWeightedMatch
from dgt.utils import graph_iterations

_logger = logging.getLogger(__name__)


def find_weight_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return 1


def clean_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        new_s = s[:start - 1] + s[end + 1:]
        return new_s
    except ValueError:
        return s


def eliminate_spaces(line):
    line = line.replace(' ', '')
    line = line.replace('\t', '')
    line = line.replace('\n', '')
    return line


class UniqueNamesModifier:
    def apply(self, g):
        from ..auxiliary import get_random_name
        substitution_dict = {}
        for v in g.vs:
            random_name = get_random_name()
            old_name = v['name']
            new_name = old_name + random_name
            v['name'] = new_name
            substitution_dict[old_name] = new_name
        try:
            for v in g.vs:
                referring_name = v['refers_to']
                if referring_name:
                    v['refers_to'] = substitution_dict[referring_name]
        except Exception as e:
            _logger.warning("Exception while substituting refers_to ID: " + str(e))
        for e in g.es:
            e['name'] += get_random_name()


class BaseForwardInference:
    def compute(self):
        return None


class ForwardInference(BaseForwardInference):
    _unique = UniqueNamesModifier()

    def __init__(self, data, knowledge, permutation_shift, max_depth=1):
        self.data = data
        self.knowledge = knowledge
        self._max_depth = max_depth
        self.permutation_shift = permutation_shift

    def __apply_clause_to_graph(self, rule, data):
        drs = data.copy()
        drs.visit(self._unique)
        w = 1

        iterations = graph_iterations(drs._g)
        drs._g = iterations[self.permutation_shift % len(iterations)]

        if not rule.gradient:
            weighted_match = GraphWeightedMatch(rule.get_hypothesis(), self.knowledge._metric,
                                                self.knowledge._relations_metric)
            w = drs.visit(weighted_match)

        is_match = drs.visit(rule)
        if not is_match:
            return drs, 0
        return drs, w

    def _compute_step(self, data_tuple):
        """
        Applies all the rules to a drs
        :return: all the variants of the drs after a rule match as a pair (<NEW_DRS>, <WEIGHT>)
        """
        data = data_tuple[0]
        prior_w = data_tuple[1]

        clauses = self.knowledge.ask_rule(data)
        results = []
        for clause_tuple in clauses:
            rule = clause_tuple[0]
            rule_weight = rule.weight
            prior_rules = list(data_tuple[2])
            if rule in prior_rules:  # A rule can be used only once per path
                continue
            drs, w = self.__apply_clause_to_graph(rule, data)
            if w > 0:
                prior_rules.append(rule)
                prior_rules.append(drs)
                results.append((drs, prior_w * w * rule_weight, prior_rules))
        return results

    def compute(self):
        results = []
        to_process = [(self.data, 1, [self.data])]
        for _ in range(self._max_depth):
            new_results = []
            for data_tuple in to_process:
                new_results += self._compute_step(data_tuple)
            if not new_results:
                break
            to_process = sorted(new_results, key=lambda x: -x[1])
            results += to_process
        results = sorted(results, key=lambda x: -x[1])
        return results


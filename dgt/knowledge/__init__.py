import re
import sys
import logging

from dgt.graph import GraphRule
from .graph_ner_cleaner import GraphNERCleaner

_logger = logging.getLogger(__name__)


def _get_list_of_rules_from_text(text):
    lines = []
    for line in text.split('\n'):
        if not line.strip() or line.strip()[0] == '#':
            continue
        lines.append(line)
    lines = '\n'.join(lines).split(';')
    return lines


def _get_substition_rule(line):
    p = re.compile('DEFINE(.*)AS(.*)')
    lst = p.findall(line)
    for item1, item2 in lst:
        return item1, item2


def _create_list_from_string(string):
    string = string.strip()
    if not string or len(string) < 2:
        return []
    string = string[1:-1]
    lst = string.split(',')
    lst = [item.strip() for item in lst]
    return lst


def _looks_like_list(string):
    string = string.strip()
    if string[0] == '[' and string[-1] == ']':
        return True
    return False


def _substitute_list_into_metric(metric, substitution):
    subst_name = substitution[0].strip()
    if _looks_like_list(substitution[1]):
        subst_list = _create_list_from_string(substitution[1])
        metric.add_substitution(subst_name, subst_list)
    return metric


def _substitute_string_into_rule(rule_str, substitution):
    subst_from = substitution[0].strip()
    subst_to = substitution[1].strip()
    rule_str = rule_str.replace(subst_from, subst_to)
    return rule_str


class Knowledge:
    def __init__(self, metric, relations_metric):
        self._rules = []
        self._metric = metric
        self._relations_metric = relations_metric
        self._substitution_list = []

    def add_graph(self, drs, sentence_number=None):
        pass

    def add_rule(self, rule, weight=1.0):
        self._rules.append((rule, weight))

    def add_rules(self, text, gradient=True):
        from ..auxiliary import LineFinder
        line_finder = LineFinder(text)
        substitution_triggers = []
        rules_lines = _get_list_of_rules_from_text(text)
        for rule_text in rules_lines:
            original_rule_text = rule_text
            if not rule_text.strip():
                continue
            for s in self._substitution_list:
                rule_text = _substitute_string_into_rule(rule_text, s)
            substitution = _get_substition_rule(rule_text)
            if substitution:
                self.metric = _substitute_list_into_metric(self._metric, substitution)
                substitution_triggers.append(substitution[0].strip())
                if not _looks_like_list(substitution[1]):
                    self._substitution_list.append(substitution)
                continue
            try:
                rule = GraphRule(rule_text, self._metric, self._relations_metric, gradient=gradient)
                rule.test()
                self.add_rule(rule)
            except SyntaxError:
                sys.stderr.write('Error in line ' + str(line_finder.get_line_number(original_rule_text)) + ':\n')
                sys.stderr.write(original_rule_text + '\n')
                sys.stderr.flush()
            finally:
                pass

    def ask_graph(self, drs):
        pass

    def ask_rule(self, drs):
        return self._rules

    def ask_rule_fw(self, drs):
        return self._rules

    def ask_rule_bw(self, drs):
        return self._rules

    def get_all_rules(self):
        return self._rules

    def copy(self):
        import copy
        to_return = Knowledge(self._metric.copy(), self._relations_metric.copy())
        to_return._rules = copy.deepcopy(self._rules)
        to_return._substitution_list = copy.deepcopy(self._substitution_list)
        return to_return
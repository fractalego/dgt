import os
import torch

from gensim.models import KeyedVectors

from parvusdb import create_graph_from_string
from parvusdb.utils.code_container import DummyCodeContainer
from parvusdb.utils.match import Match
from parvusdb.utils.match import MatchException

from dgt.graph import GraphRule
from dgt.graph.graph import Graph
from dgt.graph.node_matcher import VectorNodeMatcher
from dgt.knowledge import Knowledge
from dgt.metric import GloveMetric
from dgt.auxiliary.config import device

_path = os.path.dirname(__file__)


def get_data_goal_knowledge_from_json(json_item, metric, relations_metric):
    fact_lst = json_item['facts']
    goal_lst = json_item['goals']
    if len(fact_lst) != len(goal_lst):
        raise RuntimeError('The number of facts and goals is not the same!')
    nongrad_rules = '; '.join(json_item['non_trainable_rules'])
    grad_rules = '; '.join(json_item['trainable_rules'])
    data = []
    goals = []
    for fact in fact_lst:
        data.append(Graph.create_from_predicates_string(fact, metric, relations_metric, gradient=False))
    for goal in goal_lst:
        goals.append(Graph.create_from_predicates_string(goal, metric, relations_metric, gradient=False))
    k = Knowledge(metric=metric, relations_metric=relations_metric)
    k.add_rules(nongrad_rules, gradient=False)
    k.add_rules(grad_rules, gradient=True)
    return data, goals, k


def get_relations_embeddings_dict_from_json(json_item, embedding_size=20):
    relations = json_item['relations']
    embeddings = torch.nn.Embedding(len(relations), embedding_size)
    vectors = [embeddings(torch.LongTensor([i]))[0].detach().numpy() for i in range(len(relations))]
    model = KeyedVectors(embedding_size)
    model.add(relations, vectors)
    return GloveMetric(model, threshold=0.9)


def print_predicates(k):
    rules = k.get_all_rules()
    print('Predicates:')
    for rule in rules:
        print(rule[0].predicates())


def print_all_the_paths(end_graph):
    for item in end_graph:
        print('---')
        print(item[1])
        [print(item.predicates()) for item in item[2]]


def print_all_the_rules_with_weights(k):
    rules = k.get_all_rules()
    print('Predicates:')
    for rule in rules:
        print(rule[0].predicates(), rule[0].weight)


def get_string_with_all_the_rules_with_weights(k):
    rules = k.get_all_rules()
    str_list = []
    for rule in rules:
        str_list.append(rule[0].predicates(print_threshold=False).strip())
    return str_list


def create_rule_matrix(len_cons, len_hyp, matrix_size):
    rule_matrix = torch.zeros([matrix_size, matrix_size]).to(device)
    for i in range(len_cons):
        for j in range(len_hyp):
            rule_matrix[i, j] = 1.
    return rule_matrix


def create_graph_list(inference_list, goal):
    graph_list = []
    rule_matrices = []
    relations_rule_matrices = []
    for item in inference_list:
        if type(item) is GraphRule:
            hyp = item.get_hypothesis()
            cons = item.get_consequence()
            graph_list.append(hyp)
            weight = item.weight.clamp(min=0, max=1.)
            rule_matrices.append(weight * create_rule_matrix(len(cons._g.vs), len(hyp._g.vs), 10))
            relations_rule_matrices.append(weight * create_rule_matrix(len(cons._g.es), len(hyp._g.es), 10))
        elif type(item) is Graph:
            graph_list.append(str(item))
    graph_list.append(str(goal))
    return graph_list, rule_matrices, relations_rule_matrices


def get_proper_vector(metric, item, key):
    vector_index = item[key]
    vector = metric.get_vector_from_index(vector_index) / metric.get_vector_from_index(vector_index).norm(2)
    return vector


def create_list_of_states(metric, graph_list, match):
    pre_match = []
    post_match = []
    post_thresholds = []
    substitutions = []
    for i in range(0, len(graph_list), 2):
        gl = create_graph_from_string(str(graph_list[i]))
        gr = create_graph_from_string(str(graph_list[i + 1]))

        try:
            matching_variables = match.get_variables_substitution_dictionaries(gl, gr)
            # print(matching_variables)
            substitutions.append(matching_variables)
            pre_items = [[item['name'], get_proper_vector(metric, item, 'vector')] for item in gl.vs]
            post_items = [[item['name'], get_proper_vector(metric, item, 'vector')] for item in gr.vs]

            len_pre_items = len(pre_items)
            len_post_items = len(post_items)

            pre_items += [['dummy', torch.zeros(300).to(device)] for _ in range(10 - len_pre_items)]
            post_items += [['dummy', torch.zeros(300).to(device)] for _ in range(10 - len_post_items)]

            pre_match.append(pre_items)
            post_match.append(post_items)

            post_thrs = [metric.get_threshold_from_index(item['vector']) for item in gr.vs]
            post_thrs += [torch.ones(1).to(device) for _ in range(10 - len(post_thrs))]
            post_thresholds.append(post_thrs)

        except MatchException:
            return [], [], [], []
    return pre_match, post_match, post_thresholds, substitutions


def create_list_of_states_for_relations(metric, graph_list, match):
    pre_match = []
    post_match = []
    post_thresholds = []
    substitutions = []
    for i in range(0, len(graph_list), 2):
        gl = create_graph_from_string(str(graph_list[i]))
        gr = create_graph_from_string(str(graph_list[i + 1]))

        try:
            matching_variables = match.get_variables_substitution_dictionaries(gl, gr)
            # print(matching_variables)
            substitutions.append(matching_variables)
            pre_items = [[item['name'], get_proper_vector(metric, item, 'rvector')] for item in gl.es]
            post_items = [[item['name'], get_proper_vector(metric, item, 'rvector')] for item in gr.es]

            len_pre_items = len(pre_items)
            len_post_items = len(post_items)

            pre_items += [['dummy', torch.zeros(20).to(device)] for _ in range(10 - len_pre_items)]
            post_items += [['dummy', torch.zeros(20).to(device)] for _ in range(10 - len_post_items)]

            pre_match.append(pre_items)
            post_match.append(post_items)

            post_thrs = [metric.get_threshold_from_index(item['rvector']) for item in gr.es]
            post_thrs += [torch.ones(1).to(device) for _ in range(10 - len(post_thrs))]
            post_thresholds.append(post_thrs)

        except MatchException:
            return [], [], [], []
    return pre_match, post_match, post_thresholds, substitutions


def create_scattering_sequence(pre_match, post_match, post_thresholds, substitutions, rule_matrices,
                               nodes_or_relations):
    scattering_matrices = []
    for i in range(len(substitutions)):
        pre_vectors = torch.stack([item[1] for item in pre_match[i]])
        post_vectors = torch.stack([item[1] for item in post_match[i]])
        post_biases = torch.stack([item[0] for item in post_thresholds[i]])

        bias_matrix = torch.ones([10, 10]).to(device)
        for i2 in range(10):
            for j2 in range(10):
                bias_matrix[i2, j2] = post_biases[j2].clamp(min=0.5, max=1) * 1

        softmax = torch.nn.Softmax()
        scattering_matrix = softmax(torch.mm(post_vectors, torch.transpose(pre_vectors, 0, 1)) - bias_matrix)
        adjacency_matrix = torch.zeros(scattering_matrix.shape).to(device)
        for k, v in substitutions[i][nodes_or_relations].items():
            l_index = [item[0] for item in pre_match[i]].index(v)
            r_index = [item[0] for item in post_match[i]].index(k)
            adjacency_matrix[l_index, r_index] = 1.
        scattering_matrix = torch.mul(adjacency_matrix, scattering_matrix)
        scattering_matrices.append(scattering_matrix)

    scattering_sequence = torch.eye(10).to(device)
    for i, scattering_matrix in enumerate(scattering_matrices):
        scattering_sequence *= scattering_matrix
        try:
            rule_matrix = rule_matrices[i]
            scattering_sequence *= rule_matrix
        except:
            pass

    return scattering_sequence


def train_a_single_path(path, goal, metric, relation_metric, no_threshold_match, threshold_match, optimizer, epochs):
    for i in range(epochs):
        graph_list, rule_matrices, relations_rule_matrices = create_graph_list(path[2], goal)

        # Skip training for paths that do not have a differentiable rule
        has_gradient_rule = False
        for it in path[2]:
            if it.gradient:
                has_gradient_rule = True
                break
        if not has_gradient_rule:
            break

        # Printing path out while training
        # print('new path:')
        # [print(it.predicates()) for it in item[2]]

        pre_match, post_match, post_thresholds, substitutions = create_list_of_states(metric, graph_list,
                                                                                      no_threshold_match)
        relations_pre_match, relations_post_match, relations_post_thresholds, substitutions \
            = create_list_of_states_for_relations(relation_metric, graph_list, no_threshold_match)

        if not substitutions:
            break
        scattering_sequence = create_scattering_sequence(pre_match, post_match, post_thresholds, substitutions,
                                                         rule_matrices, nodes_or_relations=0)
        relations_scattering_sequence = create_scattering_sequence(relations_pre_match, relations_post_match,
                                                                   relations_post_thresholds, substitutions,
                                                                   relations_rule_matrices, nodes_or_relations=1)
        initial_vector = torch.ones(10).to(device)
        final_vector = torch.mv(scattering_sequence, initial_vector)
        goal_vector = torch.Tensor([0 if item[0] is 'dummy' else 1 for item in post_match[-1]]).to(device)

        relations_initial_vector = torch.ones(10).to(device)
        relations_final_vector = torch.mv(relations_scattering_sequence, relations_initial_vector)
        relations_goal_vector = torch.Tensor([0 if item[0] is 'dummy' else 1 for item in relations_post_match[-1]]).to(device)

        loss = -torch.mean(
            goal_vector * torch.log(final_vector + 1e-15) + (1 - goal_vector) * torch.log(1 - final_vector + 1e-15)
            + relations_goal_vector * torch.log(relations_final_vector + 1e-15) + (
                    1 - relations_goal_vector) * torch.log(
                1 - relations_final_vector + 1e-15)
        )

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Check if the trained sequence of rules actually satisfy the goal
        new_graph_list, _, _ = create_graph_list(path[2], goal)
        _, _, _, substitutions = create_list_of_states(metric, new_graph_list, threshold_match)
        if substitutions:
            return True

    return False


def train_all_paths(metric, relations_metric, k, paths, goal, epochs=50, step=1e-2):
    no_threshold_match = Match(matching_code_container=DummyCodeContainer(),
                               node_matcher=VectorNodeMatcher(metric, relations_metric, gradient=True))
    threshold_match = Match(matching_code_container=DummyCodeContainer(),
                            node_matcher=VectorNodeMatcher(metric, relations_metric, gradient=False))

    finished_paths = []

    for item in paths:
        vectors_to_modify = metric.get_indexed_vectors()
        threshold_to_modify = metric.get_indexed_thresholds()
        rules_weights_to_modify = [rule[0].weight for rule in k.get_all_rules()]
        relations_vectors_to_modify = relations_metric.get_indexed_vectors()
        relations_threshold_to_modify = relations_metric.get_indexed_thresholds()

        optimizer = torch.optim.Adam(vectors_to_modify + threshold_to_modify + rules_weights_to_modify
                                     + relations_threshold_to_modify + relations_vectors_to_modify,
                                     lr=step)
        if train_a_single_path(item, goal, metric, relations_metric, no_threshold_match, threshold_match, optimizer,
                               epochs):
            finished_paths.append(item)

    for item in finished_paths:
        vectors_to_modify = metric.get_indexed_vectors()
        threshold_to_modify = metric.get_indexed_thresholds()
        rules_weights_to_modify = [rule[0].weight for rule in k.get_all_rules()]

        optimizer = torch.optim.Adam(vectors_to_modify + threshold_to_modify + rules_weights_to_modify,
                                     lr=step)
        train_a_single_path(item, goal, metric, relations_metric, no_threshold_match, threshold_match, optimizer,
                            epochs)
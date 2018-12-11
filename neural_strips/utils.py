import torch

from parvusdb import create_graph_from_string
from parvusdb.utils.code_container import DummyCodeContainer
from parvusdb.utils.match import Match
from parvusdb.utils.match import MatchException

from neural_strips.drt import DrsRule
from neural_strips.drt.drs import Drs
from neural_strips.drt.node_matcher import VectorNodeMatcher
from neural_strips.knowledge import Knowledge


def get_data_goal_knowledge_from_json(json_item, metric):
    fact_lst = json_item['facts']
    goal_lst = json_item['goals']
    if len(fact_lst) != len(goal_lst):
        raise RuntimeError('The number of facts and goals is not the same!')
    nongrad_rules = '; '.join(json_item['non_trainable_rules'])
    grad_rules = '; '.join(json_item['trainable_rules'])
    data = []
    goals = []
    for fact in fact_lst:
        data.append(Drs.create_from_predicates_string(fact, metric, gradient=False))
    for goal in goal_lst:
        goals.append(Drs.create_from_predicates_string(goal, metric, gradient=False))
    k = Knowledge(metric=metric)
    k.add_rules(nongrad_rules, gradient=False)
    k.add_rules(grad_rules, gradient=True)
    return data, goals, k


def print_predicates(k):
    rules = k.get_all_rules()
    print('Predicates:')
    for rule in rules:
        print(rule[0].predicates())


def print_all_the_paths(end_drs):
    for item in end_drs:
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
    rule_matrix = torch.zeros([matrix_size, matrix_size])
    for i in range(len_cons):
        for j in range(len_hyp):
            rule_matrix[i, j] = 1.
    return rule_matrix


def create_drs_list(inference_list, goal):
    drs_list = []
    rule_matrices = []
    for item in inference_list:
        if type(item) is DrsRule:
            hyp = item.get_hypothesis()
            cons = item.get_consequence()
            drs_list.append(hyp)
            weight = item.weight.clamp(max=1.)
            rule_matrices.append(weight * create_rule_matrix(len(cons._g.vs), len(hyp._g.vs), 10))
        elif type(item) is Drs:
            drs_list.append(str(item))
    drs_list.append(str(goal))
    return drs_list, rule_matrices


def get_proper_vector(metric, item):
    vector_index = item['vector']
    vector = metric.get_vector_from_index(vector_index) / metric.get_vector_from_index(vector_index).norm(2)
    return vector


def create_list_of_states(metric, drs_list, match):
    pre_match = []
    post_match = []
    post_thresholds = []
    substitutions = []
    for i in range(0, len(drs_list), 2):
        gl = create_graph_from_string(str(drs_list[i]))
        gr = create_graph_from_string(str(drs_list[i + 1]))

        try:
            matching_variables = match.get_variables_substitution_dictionaries(gl, gr)

            substitutions.append(matching_variables)
            pre_items = [[item['name'], get_proper_vector(metric, item)] for item in gl.vs]
            post_items = [[item['name'], get_proper_vector(metric, item)] for item in gr.vs]

            len_pre_items = len(pre_items)
            len_post_items = len(post_items)

            pre_items += [['dummy', torch.zeros(300)] for _ in range(10 - len_pre_items)]
            post_items += [['dummy', torch.zeros(300)] for _ in range(10 - len_post_items)]

            pre_match.append(pre_items)
            post_match.append(post_items)

            post_thrs = [metric.get_threshold_from_index(item['vector']) for item in gr.vs]
            post_thrs += [torch.ones(1) for _ in range(10 - len(post_thrs))]
            post_thresholds.append(post_thrs)

        except MatchException:
            return [], [], [], []
    return pre_match, post_match, post_thresholds, substitutions


def create_scattering_sequence(pre_match, post_match, post_thresholds, substitutions, rule_matrices):
    scattering_matrices = []
    for i in range(len(substitutions)):
        pre_vectors = torch.stack([item[1] for item in pre_match[i]])
        post_vectors = torch.stack([item[1] for item in post_match[i]])
        post_biases = torch.stack([item[0] for item in post_thresholds[i]])

        bias_matrix = torch.ones([10, 10])
        for i2 in range(10):
            for j2 in range(10):
                bias_matrix[i2, j2] = post_biases[j2].clamp(min=0, max=1) * 1

        softmax = torch.nn.Softmax()
        scattering_matrix = softmax(torch.mm(post_vectors, torch.transpose(pre_vectors, 0, 1)) - bias_matrix)
        adjacency_matrix = torch.zeros(scattering_matrix.shape)
        for k, v in substitutions[i][0].items():
            l_index = [item[0] for item in pre_match[i]].index(v)
            r_index = [item[0] for item in post_match[i]].index(k)
            adjacency_matrix[l_index, r_index] = 1.
        scattering_matrix = torch.mul(adjacency_matrix, scattering_matrix)
        scattering_matrices.append(scattering_matrix)

    scattering_sequence = torch.eye(10)
    for i, scattering_matrix in enumerate(scattering_matrices):
        scattering_sequence *= scattering_matrix
        try:
            rule_matrix = rule_matrices[i]
            scattering_sequence *= rule_matrix
        except:
            pass

    return scattering_sequence


def train_all_paths(metric, k, paths, goal):
    no_threshold_match = Match(matching_code_container=DummyCodeContainer(),
                               node_matcher=VectorNodeMatcher(metric, gradient=True))
    threshold_match = Match(matching_code_container=DummyCodeContainer(),
                            node_matcher=VectorNodeMatcher(metric, gradient=False))
    for item in paths:
        vectors_to_modify = metric.get_indexed_vectors()
        threshold_to_modify = metric.get_indexed_thresholds()
        rules_weights_to_modify = [rule[0].weight for rule in k.get_all_rules()]

        optimizer = torch.optim.Adam(vectors_to_modify + threshold_to_modify + rules_weights_to_modify,
                                     lr=1e-2)
        for i in range(200):
            drs_list, rule_matrices = create_drs_list(item[2], goal)

            # Skip training for paths that do not have a differentiable rule
            has_gradient_rule = False
            for it in item[2]:
                if it.gradient:
                    has_gradient_rule = True
                    break
            if not has_gradient_rule:
                break

            # Printing path out while training
            # [print(it.predicates()) for it in item[2]]

            pre_match, post_match, post_thresholds, substitutions = create_list_of_states(metric, drs_list,
                                                                                          no_threshold_match)
            if not substitutions:
                break
            scattering_sequence = create_scattering_sequence(pre_match, post_match, post_thresholds, substitutions,
                                                             rule_matrices)
            initial_vector = torch.ones(10)
            final_vector = torch.mv(scattering_sequence, initial_vector)
            goal_vector = torch.Tensor([0 if item[0] is 'dummy' else 1 for item in post_match[-1]])

            loss = -torch.mean(
                goal_vector * torch.log(final_vector + 1e-15) + (1 - goal_vector) * torch.log(1 - final_vector + 1e-15))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Check if the trained sequence of rules actually satisfy the goal
            new_drs_list, _ = create_drs_list(item[2], goal)
            _, _, _, substitutions = create_list_of_states(metric, new_drs_list, threshold_match)
            if substitutions:
                break

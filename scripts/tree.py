class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', distance_from_root=0, children=None):
        self.name = name
        self.children = []
        self.parent = None
        self.distance_from_root = distance_from_root
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
        node.parent = self

# build the tree 3 2 4 4 3 2
classifier2 = Tree('classifier2', 2, [Tree(7, 3), Tree(9, 3), Tree(10, 3), Tree(11, 3)])
classifier1 = Tree('classifier1', 1, [Tree(12, 2), classifier2])

classifier5 = Tree('classifier5', 3, [Tree(2, 4), Tree(4, 4), Tree(5, 4), Tree(8, 4)])
classifier6 = Tree('classifier6', 3, [Tree(0, 4), Tree(3, 4), Tree(6, 4)])
classifier4 = Tree('classifier4', 2, [classifier5, classifier6])
classifier3 = Tree('classifier3', 1, [Tree(1, 2), classifier4, Tree(13, 2)])

classifier0 = Tree('classifier0', 0, [classifier1, classifier3])
def print_parents_to_kids(classifier0):
    root_list = []
    root_list.append(classifier0)
    result_dict = {}
    while root_list:
        curr = root_list.pop(0)
        children_list = []
        for i in curr.children:
            children_list.append(i.name)
        result_dict[curr.name] = children_list
        root_list.extend(curr.children)
    print(result_dict)
def print_kids_to_parents(classifier0):
    root_list = []
    root_list.append(classifier0)
    result_dict = {}
    while root_list:
        curr = root_list.pop(0)
        for i in curr.children:
            result_dict[i.name] = curr.name
        root_list.extend(curr.children)
    print(result_dict)
# print_parents_to_kids(classifier0)
# print_kids_to_parents(classifier0)

def findLCA(root, n1, n2):
    if root.name == n1 or root.name == n2:
        return root

    total_not_nulls = 0
    possible_candi = None
    for child in root.children:
        curr = findLCA(child, n1, n2)
        if curr:
            total_not_nulls += 1
            possible_candi = curr
    if total_not_nulls == 2: return root

    return possible_candi

node_to_distance = {}
def build_node_to_distance(node_to_distance, root):
    node_to_distance[root.name] = root.distance_from_root
    for child in root.children:
        build_node_to_distance(node_to_distance, child)
build_node_to_distance(node_to_distance, classifier0)
print(node_to_distance)

def calculate_F1(gold, our_prediction):
    LCA = findLCA(classifier0, gold, our_prediction)
    if node_to_distance[our_prediction] == 0:
        # should not happen after + 1
        assert False
        return 0
    precision = (LCA.distance_from_root + 1) / (node_to_distance[our_prediction] + 1)
    recall = (LCA.distance_from_root + 1) / (node_to_distance[gold] + 1)
    if precision + recall == 0:
        # should not happen after + 1
        assert False
        return 0
    F1 = 2 * precision * recall / (precision + recall)
    return F1
# pair_to_f1 = {}
# for i in range(14):
#     result = ""
#     for j in range(14):
#         pair_to_f1[(i, j)] = calculate_F1(i,j)
#         result += " " + str(calculate_F1(i,j))
#     print(result)
# print(pair_to_f1)



import pandas as pd
# our_file = "../outputs/no-early-stopping-together.txt"
# our_file = "../outputs/together-0-test.txt"
gold_file = "../split_data/gold_label.txt"
# our_file = "../outputs/together-0-test1.txt"

# our_file = "../outputs/test_span.txt"
# our_file = "../outputs/test_hier_1.txt"
# our_file = "../outputs/test_chunk.txt"
our_file = "../outputs/test_span_hier.txt"



# gold_file = "../outputs/together-0-test.txt"
# gold_file = "../split_data/datasets/6/dev-task-flc-tc.labels.txt"

# note only be able to handle 2 predictions case
def handle_multiple_predictions(our_technique, gold_technique):
    merged_set = set(our_technique + gold_technique)
    if len(merged_set) == 2:
        # both predictions are matched
        return 2, 0
    if len(merged_set) == 3:
        # only 1 prediction is 
        if our_technique[0] in gold_technique:
            # matched element is our_technique[0]
            gold_technique.remove(our_technique[0])
            print(gold_technique[0])
            print(our_technique[1])
            return 1, calculate_F1(gold_technique[0], our_technique[1])
        else:
            assert our_technique[1] in gold_technique
            gold_technique.remove(our_technique[1])
            print(gold_technique[0])
            print(our_technique[0])
            return 1, calculate_F1(gold_technique[0], our_technique[0])
    if len(merged_set) == 4:
        possible_f1_1 = calculate_F1(gold_technique[0], our_technique[0]) + calculate_F1(gold_technique[1], our_technique[1])
        possible_f1_2 = calculate_F1(gold_technique[0], our_technique[1]) + calculate_F1(gold_technique[1], our_technique[0])
        return 0, max(possible_f1_1, possible_f1_2)
    assert False # could only happen three situations mentioned above


def process(our_file, gold_file):
    our = pd.read_csv(our_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end", "prob"])
    gold = pd.read_csv(gold_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
    our = our.groupby(['article_id','start','end'])['techniques'].apply(list).reset_index()
    gold = gold.groupby(['article_id','start','end'])['techniques'].apply(list).reset_index()

    our_technique = our['techniques']
    gold_technique = gold['techniques']

    for i in range(len(our_technique)):
        # if gold['start'][i] == gold['start'][i + 1] and gold['end'][i] == gold['end'][i + 1] and gold['techniques'][i] != our['techniques'][i]: 
        #     print(i)
        #     assert False
        # if i >= 2 and gold['start'][i] == gold['start'][i - 1] and gold['end'][i] == gold['end'][i - 1] and gold['start'][i - 1] == gold['start'][i - 2] and gold['end'][i - 1] == gold['end'][i - 2]:
        #     assert False

        assert gold['start'][i] == our['start'][i]
        assert gold['end'][i] == our['end'][i]
        assert gold['article_id'][i] == our['article_id'][i]
    print("all correct")

    f1_sum = 0
    num = 0
    correct = 0
    prob_sum_incorrect_list = []
    prob_sum_correct_list = []
    prob_sum_incorrect = 0
    prob_sum_correct = 0
    for i in range(len(our_technique)):
        if len(our_technique[i]) != 1:
            print(our_technique[i])
            print(gold_technique[i])
            
            number_correct, incorrect_f1_sum = handle_multiple_predictions(our_technique[i], gold_technique[i])
            print(number_correct)
            print(incorrect_f1_sum)
            print("--")
            correct += number_correct
            number_incorrect = 2 - number_correct
            num += number_incorrect 
            f1_sum += incorrect_f1_sum

        else:
            # only one prediction
            try:
                our_curr = int(our_technique[i][0])
            except ValueError:
                our_curr = our_technique[i][0]
            try:
                gold_curr = int(gold_technique[i][0])
            except ValueError: # impossible, as gold should always be valid
                gold_curr = gold_technique[i][0]

            if gold_curr != our_curr:
                # prob_sum_incorrect_list.append(our['prob'][i])
                # prob_sum_incorrect += our['prob'][i]
                f1_sum += calculate_F1(gold_curr, our_curr)
                num += 1
            else:
                # prob_sum_correct_list.append(our['prob'][i])
                # prob_sum_correct += our['prob'][i]
                correct += 1
    # print(correct)
    print(num)
    # print("--")
    # print(our_file)
    # print(f"f1 sum for incorrect:{f1_sum}")
    # print(f"num of incorrect:{num}")
    # print(f"f1_sum/ num:{f1_sum/ num}")
    print(f1_sum / num)

    print((f1_sum + correct) / (num + correct))

    # print(prob_sum_incorrect / num)
    # print(prob_sum_correct / correct)
# threshold_list = [x * 0.1 for x in range(0, 11)]
# for i in threshold_list:
#     our_file = f"../outputs/together-{i}.txt"
#     gold_file = "../split_data/gold_label.txt"
#     process(our_file, gold_file)
import numpy as np
process(our_file, gold_file)

# "../outputs/test_span.txt" 0.5119062419062362, 0.8544067429966453

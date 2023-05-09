import sys
sys.path.append("models/")

import model_eval
import pandas as pd
from collections import Counter
our_file = "outputs/baseline-output-TC-1-60.txt"
gold_file = "split_data/datasets/1/dev-task-flc-tc.labels.txt"

# our_file = "outputs/baseline-output-TC-0-32.txt"
# gold_file = "evaluation/gold/dev-task-flc-tc.labels.txt"
# f1, precision, recall, f1_per_class, precision_per_class, recall_per_class = model_eval.score(our_file, gold_file)
# print(f1_per_class)
# print("precision------")
# print(precision_per_class)
# print("recall------")
# print(recall_per_class)

# gold = pd.read_csv(gold_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
# our = pd.read_csv(our_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
# gold_technique = gold['techniques']
# our_technique = our['techniques']
# wrongly_predicted = []

# for i in range(len(our_technique)):
#     if gold_technique[i] != our_technique[i]:
#         wrongly_predicted.append((gold_technique[i], our_technique[i]))

# counter = Counter(wrongly_predicted)
# error_sum = 0
# for i in range(14):
#     curr = ""
#     for j in range(14):
#         curr += str(counter[(i, j)]) + " "
#         error_sum += counter[(i, j)]
#     print(curr)
# print("______")
# print(error_sum)

# def print_out_article(article_id, start, end):
#     f = open(f"split_data/datasets/articles/article_{article_id}", "r")

#     text = f.read()
#     f.close()
#     print(text[(start - 200):(end + 200)])
#     print(text[start:end])
# test = 0
# for i in range(len(our_technique)):
#     if gold_technique[i] == 1 and our_technique[i] == 0:
#         # print(gold.iloc[i]['article_id'])
#         # print(gold.iloc[i]['start'])
#         # print(gold.iloc[i]['end'])
#         # print(our.iloc[i])
#         print("-----")
#         print(gold.iloc[i]['article_id'])
#         print_out_article(gold.iloc[i]['article_id'], gold.iloc[i]['start'], gold.iloc[i]['end'])
#         test += 1
# print(test)

# # 1, 0 - 34: hard to discern 
# # 0, 1 - 17: hard to discern 
# # 0, 3 - 15: Loaded language v.s Exaggeration, minimization: some yes(!), some no: sin so grave that it cries out to heaven for vengeance; 
# # 2, 0 - 14: Repetition v.s. Loaded language: some yes(!), some no
# # 2, 1 - 22: Repetition v.s. Name calling, labeling: yes(! Molester in article 49), no(non-american in article 31)
# # 3, 0 - 24: Exaggeration, minimization v.s. Loaded language: some yes(!), some no: crocodile tears
# print(34 + 17 + 15 + 14 + 22 + 24)
# print(126/386)
# print(386/1302)
# #1302


# pred_span-75-0.9_0!.txt
our_file = "outputs/pred_span-75-0.9_0.9!.txt"
gold_file = "split_data/gold_label.txt"
our = pd.read_csv(our_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
gold = pd.read_csv(gold_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
our_technique = our['techniques']
gold_technique = gold['techniques']
f1, precision, recall, f1_per_class, precision_per_class, recall_per_class = model_eval.score(our_file, gold_file)
print(f1_per_class)
print("precision------")
print(precision_per_class)
print("recall------")
print(recall_per_class)

# from sklearn.metrics import f1_score
# test = f1_score(gold_technique, our_technique, average=None)
# print(test)
# print(sum(test)/14)

pair_list = []
for i in range(len(our_technique)):
    pair_list.append((gold_technique[i], our_technique[i]))
counter = Counter(pair_list)
print(len(gold))
print(len(our))
print(our_file)
def calculate_new_f1(gold_labels):
    rest = [i for i in range(14) if i not in gold_labels]
    def helper(list1, list2, counter):
        ret = 0
        for i in list1:
            for j in list2:
                ret += counter[i, j]
        return ret
    # calculate TP
    TP = helper(gold_labels, gold_labels, counter)
    FP = helper(rest, gold_labels, counter)
    TN = helper(rest, rest, counter)
    FN = helper(gold_labels, rest, counter)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

precision, recall, f1 = calculate_new_f1([7, 9, 10, 11])
print(f1)
precision, recall, f1 = calculate_new_f1([7, 9, 10, 11, 12])
print(f1)
precision, recall, f1 = calculate_new_f1([2, 4, 5, 8])
print(f1)
precision, recall, f1 = calculate_new_f1([0, 3, 6])
print(f1)
precision, recall, f1 = calculate_new_f1([2, 4, 5, 8, 0, 3, 6])
print(f1)
precision, recall, f1 = calculate_new_f1([2, 4, 5, 8, 0, 3, 6, 1, 13])
print(f1)

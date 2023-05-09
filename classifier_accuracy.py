import sys
sys.path.append("models/")

import model_eval
import pandas as pd
from collections import Counter

our_file = "outputs/pred_span-75-1_1!.txt"
gold_file = "split_data/gold_label.txt"
our = pd.read_csv(our_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
gold = pd.read_csv(gold_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
our_technique = our['techniques']
gold_technique = gold['techniques']
# sort the two files 
f1, precision, recall, f1_per_class, precision_per_class, recall_per_class = model_eval.score(our_file, gold_file)
print(f1_per_class)
print("precision------")
print(precision_per_class)
print("recall------")
print(recall_per_class)

pair_list = []
for i in range(len(our_technique)):
    pair_list.append((gold_technique[i], our_technique[i]))
counter = Counter(pair_list)
print(len(gold))
print(len(our))
print(our_file)
def calculate_new_accuracy(gold_labels):
    rest = [i for i in range(14) if i not in gold_labels]
    def helper1(gold_labels, counter):
        ret = 0
        for i in gold_labels:
            ret += counter[i, i]
        return ret
    def helper2(gold_labels, counter):
        ret = 0
        for i in gold_labels:
            for j in range(14):
                ret += counter[i, j]
        return ret
    # calculate TP
    correct = helper1(gold_labels, counter)
    all_ = helper2(gold_labels, counter)
    
    return correct/ all_
print("classifier2 - 4")
print(calculate_new_accuracy([7, 9, 10, 11]))
print("classifier5 - 4")
print(calculate_new_accuracy([2, 4, 5, 8]))

print("classifier6 - 3")
print(calculate_new_accuracy([0, 3, 6]))
print("classifier3 - 3")
print(calculate_new_accuracy([2, 4, 5, 8, 0, 3, 6, 1, 13]))

print("classifier1 - 2")
print(calculate_new_accuracy([7, 9, 10, 11, 12]))
print("classifier4 - 2")
print(calculate_new_accuracy([2, 4, 5, 8, 0, 3, 6]))
print("classifier0 - 2")
print(calculate_new_accuracy([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))

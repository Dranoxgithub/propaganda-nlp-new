import pandas as pd
import collections
import math
import pickle



inputFileName = '../datasets/train-task-flc-tc.labels'
outputFileName = '../datasets/subtrain-task-flc-tc.labels'
df = pd.read_csv(inputFileName, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])

print(len(df))
df = df.sample(n=1063, replace=False, random_state=1)
print(len(df))

df.to_csv(outputFileName, header=None, index=None, sep='\t', mode='w')
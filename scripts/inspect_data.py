import pandas as pd
import collections
import math
import pickle

df = pd.read_csv('/usr/xtmp/ac638/propaganda-nlp/split_data/full', sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])

mapping = {'Loaded_Language': 0, 'Name_Calling,Labeling': 1, "Repetition": 2, "Exaggeration,Minimisation": 3,
"Doubt": 4, "Appeal_to_fear-prejudice": 5, "Flag-Waving": 6, "Causal_Oversimplification": 7,
"Slogans": 8, "Appeal_to_Authority": 9, "Black-and-White_Fallacy": 10, "Thought-terminating_Cliches": 11,
"Whataboutism,Straw_Men,Red_Herring": 12, "Bandwagon,Reductio_ad_hitlerum": 13}


df.replace({'techniques': mapping}, inplace=True)
print(df[10])
article = df.loc[df['techniques'] == 1]
print(len(article))

# # df.to_csv("/usr/xtmp/ac638/propaganda-nlp/evaluation/gold/dev-task-flc-tc.labels.txt", header=None, index=None, sep='\t', mode='w')
# df.sort_values(by=['end'], ascending=False, inplace=True)
# df.sort_values(by=['article_id', 'start'], inplace=True)
# article_set = set(df['article_id'])

# for article_id in article_set:
#     article = df.loc[df['article_id'] == article_id]
#     print(article)
#     for i in article.iterrows():
#         print(i['start'])
#     break



# print(len(set(df['article_id'])))
# print(len(set(df_1['article_id'])))
# test = set(df['article_id'])
# for i in set(df_1['article_id']):
#     if i not in test:
#         print(i)
# counts =[]
# for j in range(14):
#     count = 0
#     for i in df['techniques']:
#         if i == j:
#             count += 1
#     counts.append(count)
#     print(f'{j} - {count}')
# print(counts)
# print(sum(counts))



# print(len([i if i == 1 for i in df['techniques']]))

# f = open("../datasets/train-articles/article111111111.txt", "r")
# text = f.read()
# print(text[265:323])
# print(text[1795:1935])

# df = pickle.load( open( "../processed_data/train_512_span.pkl", "rb" ) )
# print(df.__dict__)
# print(df['512_span_tokens'][1][2])

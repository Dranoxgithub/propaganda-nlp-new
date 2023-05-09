import pandas as pd
import pickle

# inputFileName = '../datasets/dev-task-flc-tc.labels'

# inputFileName = '../datasets/test-task-tc-template.out'
# inputFileName = '../datasets/test_format.txt'

# outputFileName = '../datasets/test_format_debug'

# inputFileName = '../datasets/test_format_duplicate.txt'

inputFileName = '../outputs/test_output_chunk-19.txt'
outputFileName = '../datasets/test_output_chunk-19-debug.txt'


# inputFileName = '../outputs/test_output_hier-20 -0.5.txt'
# outputFileName = '../outputs/test_output_hier-20-0.5.txt'
####


# df = pd.read_csv(inputFileName, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])

df = pd.read_csv(inputFileName, sep = "\t", header=None, names=['article_id', 'prediction', "start", "end"])


# mapping = {'Loaded_Language': 1, 'Name_Calling,Labeling': 2, "Repetition": 3, "Exaggeration,Minimisation":4,
# "Doubt":5, "Appeal_to_fear-prejudice": 6, "Flag-Waving": 7, "Causal_Oversimplification": 8,
# "Slogans": 9, "Appeal_to_Authority": 10, "Black-and-White_Fallacy": 11, "Thought-terminating_Cliches": 12,
# "Whataboutism,Straw_Men,Red_Herring": 13, "Bandwagon,Reductio_ad_hitlerum": 14}

# mapping = {'Loaded_Language': 0, 'Name_Calling,Labeling': 1, "Repetition": 2, "Exaggeration,Minimisation":3,
# "Doubt":4, "Appeal_to_fear-prejudice": 5, "Flag-Waving": 6, "Causal_Oversimplification": 7,
# "Slogans": 8, "Appeal_to_Authority": 9, "Black-and-White_Fallacy": 10, "Thought-terminating_Cliches": 11,
# "Whataboutism,Straw_Men,Red_Herring": 12, "Bandwagon,Reductio_ad_hitlerum": 13}



# df.replace({'techniques': mapping}, inplace=True)
# df.sort_values(by=['end'], ascending=False, inplace=True)
# df.sort_values(by=['article_id', 'start'], inplace=True)


# set duplicated entry to have technique 1, 2, 3.. instead of 0 for the test case
df['techniques'] = 0
technique_num = 1
while len(df.loc[df.duplicated(subset=['article_id', 'start', 'end', 'techniques'])]) != 0:
    df.loc[df.duplicated(subset=['article_id', 'start', 'end', 'techniques']), 'techniques'] = technique_num
    technique_num += 1
    print(technique_num)
assert len(df.loc[df.duplicated(subset=['article_id', 'start', 'end', 'techniques'])]) == 0

article = df.loc[df['article_id'] == 813552066]
print(article)
# pickle.dump(df, open(outputFileName, "wb"))

df.to_csv(outputFileName, sep = "\t", index = False, header=None)







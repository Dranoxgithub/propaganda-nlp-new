import pandas as pd
import pickle

# inputFileName = '../datasets/dev-task-flc-tc.labels'

# inputFileName = '../outputs/test_output_chunk-19.txt'
# outputFileName = '../outputs/test_output_chunk-19-test.txt'



# model_num = 0
# weight = 1
# inputFileName = f"../outputs/test_output_hier-{model_num}-{weight}.txt"
# outputFileName = f"../outputs/test_output_hier-{model_num}-{weight}-test.txt"

inputFileName = '../outputs/test_output_chunk-1.txt'
outputFileName = '../outputs/test_output_chunk-2.txt'

formatFileName = '../datasets/test_format_duplicate1.txt'



df = pd.read_csv(inputFileName, sep = "\t", header=None, names=['article_id', 'prediction', "start", "end"])


df['matching_index'] = 0
matching_index_num = 1
while len(df.loc[df.duplicated(subset=['article_id', 'start', 'end', 'matching_index'])]) != 0:
    df.loc[df.duplicated(subset=['article_id', 'start', 'end', 'matching_index']), 'matching_index'] = matching_index_num
    matching_index_num += 1
assert len(df.loc[df.duplicated(subset=['article_id', 'start', 'end', 'matching_index'])]) == 0


format_df = pd.read_csv(formatFileName, sep = "\t", header=None, names=['article_id', 'matching_index', "start", "end"])

# mapping = {'Loaded_Language': 1, 'Name_Calling,Labeling': 2, "Repetition": 3, "Exaggeration,Minimisation":4,
# "Doubt":5, "Appeal_to_fear-prejudice": 6, "Flag-Waving": 7, "Causal_Oversimplification": 8,
# "Slogans": 9, "Appeal_to_Authority": 10, "Black-and-White_Fallacy": 11, "Thought-terminating_Cliches": 12,
# "Whataboutism,Straw_Men,Red_Herring": 13, "Bandwagon,Reductio_ad_hitlerum": 14}

mapping = {'Loaded_Language': 0, 'Name_Calling,Labeling': 1, "Repetition": 2, "Exaggeration,Minimisation":3,
"Doubt":4, "Appeal_to_fear-prejudice": 5, "Flag-Waving": 6, "Causal_Oversimplification": 7,
"Slogans": 8, "Appeal_to_Authority": 9, "Black-and-White_Fallacy": 10, "Thought-terminating_Cliches": 11,
"Whataboutism,Straw_Men,Red_Herring": 12, "Bandwagon,Reductio_ad_hitlerum": 13}



# df.replace({'techniques': mapping}, inplace=True)
# df.sort_values(by=['end'], ascending=False, inplace=True)
# df.sort_values(by=['article_id', 'start'], inplace=True)


decode_mapping = {0: 'Loaded_Language', 1: 'Name_Calling,Labeling', 2: "Repetition", 3:"Exaggeration,Minimisation",
4:"Doubt", 5:"Appeal_to_fear-prejudice", 6:"Flag-Waving", 7:"Causal_Oversimplification",
8:"Slogans", 9:"Appeal_to_Authority", 10:"Black-and-White_Fallacy", 11:"Thought-terminating_Cliches",
12:"Whataboutism,Straw_Men,Red_Herring", 13:"Bandwagon,Reductio_ad_hitlerum"}



# decode
# decode_mapping = {}
# for key, value in enumerate(mapping):
#     decode_mapping[key] = value




# df['article_id']=format_df['article_id'].astype(int) 
# df['start']=format_df['start'].astype(int) 
# df['end']=format_df['end'].astype(int) 
# df['matching_index']=format_df['matching_index'].astype(int) 
format_df['prediction'] = 0
for index,row in format_df.iterrows():
    curr_row = df.loc[(df['article_id'] == row['article_id']) & (df['start'] == row['start']) & (df['end'] == row['end']) & (df['matching_index'] == row['matching_index'])]
    assert len(curr_row) == 1
    format_df.at[index, 'prediction'] = curr_row['prediction']
    # print(curr_row['prediction'])
print(len(format_df))

format_df = format_df.drop(['matching_index'], axis = 1)




# print(decode_mapping)
format_df.replace({'prediction': decode_mapping}, inplace=True)
print(df.head(10))

article = format_df.loc[format_df['article_id'] == 833041409]
print(article)


format_df.to_csv(outputFileName, sep = "\t", index = False, header=None, columns=["article_id","prediction","start", "end"])


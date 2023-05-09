import pickle
from transformers import LongformerTokenizer

train_or_dev = "dev" # dev or train 
context_length = "long"
def processArticle(article_id, article, df, tokenizer):
    f = open(f"../datasets/{train_or_dev}-articles/article{article_id}.txt", "r")
    text = f.read()
    f.close()
    number_truncated = 0
    for index,row in article.iterrows():
        curr_string = text[row['start']: row['end']]
        new_text_list = text.split(curr_string)
        new_text  = new_text_list[0] + "<bop>" + curr_string + "<eop>" + new_text_list[1]
        
        tokenized_new_text = tokenizer(new_text)['input_ids']


        df.at[index, 'phrase'] = curr_string
        df.at[index, f'{context_length}_span_tokens'] = tokenized_new_text
        bop_index = tokenized_new_text.index(50265)
        df.at[index, 'bop_index'] = bop_index
    return number_truncated

df = pickle.load( open( f"../processed_data/{train_or_dev}.pkl", "rb" ) )
df = df.astype('object')
df = df.groupby(['article_id','start','end'])['techniques'].apply(list).reset_index()	
df['one_hot'] = df.apply(lambda x: [1 if i in x['techniques'] else 0 for i in range(14)], axis=1)
# find the list of articles that we need  
article_set = set(df['article_id'])
number_processed_articles = 0

df['phrase'] = ""
df[f'{context_length}_span_tokens'] = [[] for _ in range(len(df))]
df[f'{context_length}_span_tokens'] = df[f'{context_length}_span_tokens'].astype('object')
df['bop_index'] = -1
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", additional_special_tokens=("<bop>", "<eop>"))
total_truncated = 0
for article_id in article_set:
    article = df.loc[df['article_id'] == article_id]
    total_truncated += processArticle(article_id, article, df, tokenizer)
    number_processed_articles += 1

# sanity check that all the bops are found
assert -1 not in list(df['bop_index']) 
print("Successfully processed", number_processed_articles)
print("Truncated", total_truncated)
print(len(df))
pickle.dump(df, open(f'../processed_data/merged_{train_or_dev}_{context_length}_span_new.pkl', "wb"))	
df.to_csv(f'../processed_data/merged_{train_or_dev}_{context_length}_span_new.csv', index = False)

# no attention mask
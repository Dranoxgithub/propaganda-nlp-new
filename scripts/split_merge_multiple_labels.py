import pickle
from transformers import RobertaTokenizer

train_or_dev = "dev" # dev or train 
curr_split = 1
context_length = 512
def processArticle(article_id, article, df, tokenizer):
    f = open(f"../split_data/datasets/articles/article_{article_id}", "r")
    text = f.read()
    f.close()
    number_truncated = 0
    for index,row in article.iterrows():
        curr_string = text[row['start']: row['end']]
        new_text_list = text.split(curr_string)
        new_text  = new_text_list[0] + "<bop>" + curr_string + "<eop>" + new_text_list[1]
        
        tokenized_new_text = tokenizer(new_text)['input_ids']
        
        bop_index = tokenized_new_text.index(50265)
        eop_index = tokenized_new_text.index(50266)
        phrase_token = eop_index - bop_index + 1
        span_in_tokens = []
        if phrase_token <= context_length:
            context_remaining = context_length - phrase_token 
            start_index = max(bop_index - int(context_remaining / 2), 0) 
            end_index = min(bop_index + context_length, len(tokenized_new_text))
            span_in_tokens = tokenized_new_text[bop_index:end_index]
        else: 
            start_index = bop_index
            end_index = min(bop_index + context_length, len(tokenized_new_text)) - 1 # to replace the last one as 50266
            test = tokenized_new_text[bop_index:end_index + 1]
            if 50265 not in test or 50266 not in test:
                if 50265 not in test: assert False
                number_truncated += 1
            span_in_tokens = tokenized_new_text[bop_index:end_index]
            span_in_tokens.append(50266)
        
        assert 50265 in span_in_tokens and 50266 in span_in_tokens

        # pad it with </s> to the same length as context_length
        padded_span_in_tokens = span_in_tokens + [2] * (context_length - len(span_in_tokens))
        attention_mask_for_token = [1] * len(span_in_tokens) + [0] * (context_length - len(span_in_tokens))
        assert len(padded_span_in_tokens) == context_length and len(attention_mask_for_token) == context_length
        df.at[index, 'phrase'] = curr_string
        df.at[index, f'{context_length}_span_tokens'] = padded_span_in_tokens
        df.at[index, 'attention_mask'] = attention_mask_for_token
        bop_index = span_in_tokens.index(50265)
        df.at[index, 'bop_index'] = bop_index
    return number_truncated

df = pickle.load( open( f"../split_data/processed_data/{curr_split}/{train_or_dev}.pkl", "rb" ) )
df = df.astype('object')
df = df.groupby(['article_id','start','end'])['techniques'].apply(list).reset_index()	
df['one_hot'] = df.apply(lambda x: [1 if i in x['techniques'] else 0 for i in range(14)], axis=1)
# find the list of articles that we need  
article_set = set(df['article_id'])
number_processed_articles = 0

df['phrase'] = ""
df['attention_mask'] = [[] for _ in range(len(df))]
df[f'{context_length}_span_tokens'] = [[] for _ in range(len(df))]
df[f'{context_length}_span_tokens'] = df[f'{context_length}_span_tokens'].astype('object')
df['bop_index'] = -1
tokenizer = RobertaTokenizer.from_pretrained("roberta-large", additional_special_tokens=("<bop>", "<eop>"))
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
# pickle.dump(df, open(f'../split_data/processed_data/{curr_split}/merged_{train_or_dev}_{context_length}_span_new.pkl', "wb"))	
# df.to_csv(f'../split_data/processed_data/{curr_split}/merged_{train_or_dev}_{context_length}_span_new.csv', index = False)
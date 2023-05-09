import pickle
from transformers import RobertaTokenizer
import pandas as pd
import csv


train_or_dev = "validation" # test / training / validation
df = pd.read_csv(f"../../datasets/climate_change_data/{train_or_dev}.csv") 

df.drop(df[df.claim == "0_0"].index, inplace=True)
df = df.astype(str)
tokenizer = RobertaTokenizer.from_pretrained("roberta-large", additional_special_tokens=("<bop>", "<eop>"))
df['tokens'] = df['text'].apply(lambda x: tokenizer(x)['input_ids'])
df['label'] = df['claim'].apply(lambda x: int(x.split("_")[0]) - 1)
df['sublabel'] = df['claim'].apply(lambda x: int(x.split("_")[1]) - 1)
df['paragraph_index'] = range(len(df))

def get_combined_label(label,sublabel):
    sublabel = int(sublabel)
    if label == 0:
        return sublabel
    elif label == 1:
        return 8 + sublabel
    elif label == 2:
        return 13 + sublabel
    elif label == 3:
        return 19 + sublabel
    elif label == 4:
        return 24 + sublabel

    assert false 


df['combined_label'] = df.apply(lambda x: get_combined_label(x.label, x.sublabel), axis=1)

pickle.dump(df, open(f'../../processed_data/climate_change_data/{train_or_dev}.pkl', "wb"))
df.to_csv(f'../../processed_data/climate_change_data/{train_or_dev}.csv', index = False)
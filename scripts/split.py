import pickle
import pandas as pd

train_or_dev = "train" # dev or train 
def processArticle(article_id, article, df):
    f = open(f"../datasets/{train_or_dev}-articles/article{article_id}.txt", "r")
    text = f.read()
    f.close()

    text = text.replace("\n", " ")
    split_num = -1
    first_split = False
    for i in range(6):
        f = open(f"../datasets/split/fold-{i + 1}", "r")
        split_text = f.read()
        f.close()
        if text[:-1] in split_text:
            assert first_split == False
            split_num = i + 1
            first_split = True

    # assert split_num != -1
    return split_num

df = pickle.load( open( f"../processed_data/{train_or_dev}.pkl", "rb" ) )
df = df.astype('object')
article_set = set(df['article_id'])

for article_id in [771879020]:
    article = df.loc[df['article_id'] == article_id]
    split_num = processArticle(article_id, article, df)
    # Append-adds at last
    # file1 = open("../processed_data/split_info.txt", "a")  # append mode
    # file1.write(f"{article_id}\t{split_num}\n")
    # file1.close()
    # if split_num == -1:
    #     print(article_id)
    print(split_num)

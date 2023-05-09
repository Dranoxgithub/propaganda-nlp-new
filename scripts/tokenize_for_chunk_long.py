import pickle
import pandas as pd
from transformers import LongformerTokenizer
import torch 

total_windows = 0 
truncated_windows = 0
train_or_dev = "train" # dev or train
split = True
curr_split = 6
context_length = 1024

def processArticle(article_id, article, df, tokenizer, new_df, save_processed_text=False, debug=False):
    if split: f = open(f"../split_data/datasets/articles/article_{article_id}", "r")
    else: f = open(f"../datasets/{train_or_dev}-articles/article{article_id}.txt", "r")
    text = f.read()
    f.close()
    original_text = text

    added_char = 0
    not_processed = []
    if debug:        
        print("--------------------------")
        print(article_id)
    for index,row in article.iterrows():
        if debug: print(f"{row['start']}: {row['end']} {text[row['start']: row['end']]} -label{row['techniques']}")
        not_processed.append(row)
    curr_start, curr_end = 0, 0
    processed_dict = {}
    shift_in_char_index = {}
    max_num_bop = 0
    for i in range(10):
        not_processed, shift_in_char_index, text, processed = insertBopAndEop(text, not_processed, f"<bop{i}>", f"<eop{i}>", shift_in_char_index, False, original_text)
        processed_dict[i] = processed 
        if len(not_processed) == 0:
            max_num_bop = max(max_num_bop, i)
            break
    if save_processed_text:
        f = open(f"../processed_data/processed_articles/{article_id}.txt", "w")
        f.write(text)
        f.close()

    tokenized_new_text = tokenizer(text)['input_ids']
    all_bop_index, bop_to_eop, bop_to_label, bop_to_start_end = tokenizeAndGetIndex(text, processed_dict, tokenized_new_text, max_num_bop)
    if debug:
        print("---------------------")
        print(all_bop_index)
        print("---------------------")
        print(bop_to_eop)
        print("---------------------")
        print(bop_to_label)
    start_index = 0
    window_size = context_length
    step_size = int(context_length / 2)
    # ensure that the start of the window always comes before a prediction
    bop_index_processed = []
    while start_index < all_bop_index[-1]:
        end_index = start_index + window_size
        bop_index_in_window_local, eop_index_in_window_local, labels, labels_in_context, labels_mask, bop_index_in_window, chunk_in_tokens, bop_to_start_in_context, bop_to_end_in_context, context_len_for_prediction, min_two_sides, global_attention_mask = truncateAndCreateDatapoints_new(tokenized_new_text, all_bop_index, bop_to_eop, bop_to_label, start_index, end_index, bop_to_start_end)
        bop_index_processed.extend(bop_index_in_window)
        # do not put inside the dataframe if there is nothing in the window
        if len(bop_index_in_window_local) == 0:
            start_index += step_size
            continue

        # pad c
        padded_chunk_in_tokens = chunk_in_tokens + [2] * (context_length - len(chunk_in_tokens))
        attention_mask_for_token = [1] * len(chunk_in_tokens) + [0] * (context_length - len(chunk_in_tokens))
        assert len(padded_chunk_in_tokens) == context_length and len(attention_mask_for_token) == context_length and len(labels_in_context) == context_length
        
        global total_windows
        total_windows += 1

        new_df = new_df.append({'bop_index' : bop_index_in_window_local, 'eop_index' : eop_index_in_window_local, 'labels' : labels, f'{context_length}_chunk_tokens' : padded_chunk_in_tokens, 
            'article_id': article_id, 'attention_mask': attention_mask_for_token, 'window_start_index' : start_index, f'labels_in_{context_length}': labels_in_context, 'labels_mask': labels_mask,
            'start_in_context': bop_to_start_in_context, 'end_in_context' : bop_to_end_in_context, 'context_len_for_prediction': context_len_for_prediction, 'min_two_sides': min_two_sides, 'global_attention_mask': global_attention_mask}, ignore_index = True)

        start_index += step_size
    return max_num_bop, new_df, bop_to_start_end, set(bop_index_processed)
def truncateAndCreateDatapoints_new(tokenized_new_text, all_bop_index, bop_to_eop, bop_to_label, window_start, window_end, bop_to_start_end, debug=False):
    chunk_in_tokens = tokenized_new_text[window_start: window_end]

    bop_index_in_window = []
    to_be_padded_bop = []
    # collect al the bops that possibly need to be padded 
    for bop_index in all_bop_index:
        if bop_index >= window_start and bop_index < window_end:
            bop_index_in_window.append(bop_index)
            if bop_to_eop[bop_index] >= window_end:
                to_be_padded_bop.append(bop_index)
        # all_bop_index is in ascending order 
        if bop_index >= window_end: break
    if len(to_be_padded_bop) != 0:
        print("start-------------------------------------------------")
        print(window_start)
        print(window_end)
        print(chunk_in_tokens[-10:])
        for i in to_be_padded_bop:
            print(f"{i} - {bop_to_eop[i]}")
        print(bop_index_in_window)
        print(bop_to_eop)
    # change chunk and change 
    removed_bop = []
    old_eop_to_new_eop, new_eop_to_old_eop, old_bop_to_new_bop, new_bop_to_old_bop = {}, {}, {}, {}
    to_be_padded_bop, bop_index_in_window, removed_bop, finish_processing = pad_check(window_start, to_be_padded_bop, bop_index_in_window, chunk_in_tokens, removed_bop, bop_to_eop, new_bop_to_old_bop, new_eop_to_old_eop)
    while not finish_processing:
        chunk_in_tokens = shift_chunk(tokenized_new_text, window_start, removed_bop, bop_to_eop)
        old_eop_to_new_eop, new_eop_to_old_eop, old_bop_to_new_bop, new_bop_to_old_bop = generate_old_eop_to_new_eop(removed_bop, bop_index_in_window, bop_to_eop)
        to_be_padded_bop, bop_index_in_window, removed_bop, finish_processing = pad_check(window_start, to_be_padded_bop, bop_index_in_window, chunk_in_tokens, removed_bop, bop_to_eop, new_bop_to_old_bop, new_eop_to_old_eop)

    # pad eop to the end
    if len(to_be_padded_bop) != 0:
        global truncated_windows
        truncated_windows += 1
        print(f"to_be_padded: {to_be_padded_bop}")
        eop_to_bop_for_padded, chunk_in_tokens = pad_eop_to_end(window_start, chunk_in_tokens, to_be_padded_bop, bop_to_eop, tokenized_new_text)
        print(chunk_in_tokens[-10:])
        print(bop_index_in_window)

    labels = []
    eop_index_in_window_local = []
    bop_index_in_window_local = []
    global_attention_mask = [0] * context_length
    # the label for padding is 14
    labels_in_context = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * context_length
    labels_mask = [0] * context_length
    bop_to_start_in_context = [-1] * context_length
    bop_to_end_in_context = [-1] * context_length
    context_len_for_prediction = [0] * context_length
    min_two_sides = [0] * context_length
    for bop_index in bop_index_in_window: # bop_index_in_window all old 
        new_bop_index = bop_index
        if bop_index in old_bop_to_new_bop: 
            new_bop_index = old_bop_to_new_bop[bop_index]
        bop_index_in_window_local.append(new_bop_index - window_start)
        global_attention_mask[new_bop_index - window_start] = 1
        print(new_bop_index - window_start)
        print(chunk_in_tokens[new_bop_index - window_start])
        assert chunk_in_tokens[new_bop_index - window_start] >= 50265
        if bop_index in to_be_padded_bop:
            eop_index_local = eop_to_bop_for_padded[bop_index] - window_start
            eop_index_in_window_local.append(eop_index_local)
            context_len_for_prediction[new_bop_index - window_start] = eop_to_bop_for_padded[bop_index] - new_bop_index
            min_two_sides[new_bop_index - window_start] = min(new_bop_index - window_start, context_length - eop_index_local - 1)
        else:
            eop_index = bop_to_eop[bop_index]
            if eop_index in old_eop_to_new_eop:
                eop_index = old_eop_to_new_eop[eop_index]
            eop_index_local = eop_index - window_start
            eop_index_in_window_local.append(eop_index_local)
            context_len_for_prediction[new_bop_index - window_start] = eop_index - new_bop_index
            min_two_sides[new_bop_index - window_start] = min(new_bop_index - window_start, context_length - eop_index_local - 1)
        labels.append(bop_to_label[bop_index])
        labels_in_context[new_bop_index - window_start] = bop_to_label[bop_index]
        labels_mask[new_bop_index - window_start] = 1
        start, end = bop_to_start_end[bop_index]
        bop_to_start_in_context[new_bop_index - window_start] = start
        bop_to_end_in_context[new_bop_index - window_start] = end

    return bop_index_in_window_local, eop_index_in_window_local, labels, labels_in_context, labels_mask, bop_index_in_window, chunk_in_tokens, bop_to_start_in_context, bop_to_end_in_context, context_len_for_prediction, min_two_sides, global_attention_mask
def pad_eop_to_end(window_start, chunk_in_tokens, to_be_padded_bop, bop_to_eop, tokenized_new_text):
    all_eops = []
    for i in to_be_padded_bop:
        all_eops.append(bop_to_eop[i])
    sorted(all_eops)

    eop_to_bop_for_padded = {}
    for i in range(len(all_eops)):
        index = len(chunk_in_tokens) - len(all_eops) + i
        curr_eop = all_eops[i]
        chunk_in_tokens[index] = tokenized_new_text[curr_eop]
        curr_bop = use_value_for_key(curr_eop, bop_to_eop)
        eop_to_bop_for_padded[curr_bop] = index + window_start
    return eop_to_bop_for_padded, chunk_in_tokens
def generate_old_eop_to_new_eop(removed, bop_index_in_window, bop_to_eop):
    removed_bop_and_eop = []
    for i in removed:
        removed_bop_and_eop.append(i)
        removed_bop_and_eop.append(bop_to_eop[i])
    old_eop_to_new_eop = {}
    new_eop_to_old_eop = {}
    old_bop_to_new_bop = {}
    new_bop_to_old_bop = {}
    # update eops
    for i in bop_index_in_window:
        shift_in_index = get_shift_in_index(i, removed_bop_and_eop)
        if shift_in_index != 0: 
            old_bop_to_new_bop[i] = i - shift_in_index
            new_bop_to_old_bop[i - shift_in_index] = i
        
        curr_eop = bop_to_eop[i]
        shift_in_index = get_shift_in_index(curr_eop, removed_bop_and_eop)
        if shift_in_index != 0: 
            old_eop_to_new_eop[curr_eop] = curr_eop - shift_in_index
            new_eop_to_old_eop[curr_eop - shift_in_index] = curr_eop
    return old_eop_to_new_eop, new_eop_to_old_eop, old_bop_to_new_bop, new_bop_to_old_bop
def get_shift_in_index(curr_eop, removed_bop_and_eop):
    removed_bop_and_eop.append(curr_eop)
    removed_bop_and_eop.sort()
    shift_in_index = removed_bop_and_eop.index(curr_eop)
    removed_bop_and_eop.remove(curr_eop)
    return shift_in_index

def shift_chunk(tokenized_new_text, window_start, removed_bop, bop_to_eop):
    # collect all indexes of the bop and eop tokens that need to be remoeved 
    removed_bop_and_eop = []
    for i in removed_bop:
        removed_bop_and_eop.append(i)
        removed_bop_and_eop.append(bop_to_eop[i])

    i  = window_start
    chunk_in_tokens = []
    while i < len(tokenized_new_text):
        if i not in removed_bop_and_eop:
            chunk_in_tokens.append(tokenized_new_text[i])
        if len(chunk_in_tokens) == context_length:
            return chunk_in_tokens
        i += 1
    return chunk_in_tokens
def pad_check(window_start, to_be_padded_bop, bop_index_in_window, chunk_in_tokens, removed, bop_to_eop, new_bop_to_old_bop, new_eop_to_old_eop):
    # check tokens that are going to be replaced
    for i in range(len(to_be_padded_bop)):
        index = len(chunk_in_tokens) - len(to_be_padded_bop) + i
        curr_token = chunk_in_tokens[index]
        # if <bopi>
        if curr_token >= 50265 and curr_token % 2 == 1:
            bop_index = window_start + index
            if bop_index in new_bop_to_old_bop: bop_index = new_bop_to_old_bop[bop_index]
            if bop_index in to_be_padded_bop: to_be_padded_bop.remove(bop_index) # remove if inside 
            if bop_index in removed: continue # case 3, hack 
            removed.append(bop_index)
            bop_index_in_window.remove(bop_index)
            return to_be_padded_bop, bop_index_in_window, removed, False
        # if <eopi> 
        if curr_token >= 50265 and curr_token % 2 == 0:
            eop_index = window_start + index
            if eop_index in new_eop_to_old_eop: eop_index = new_eop_to_old_eop[eop_index]
            bop_index = use_value_for_key(eop_index, bop_to_eop)
            if bop_index in to_be_padded_bop: to_be_padded_bop.remove(bop_index) # remove if inside 
            removed.append(bop_index)
            if bop_index in bop_index_in_window: 
                bop_index_in_window.remove(bop_index)
            else: # this occurs when the corresponding bop is before the current window, safe to pad
                continue
            # change chunk_in_tokens and bop tokens 
            return to_be_padded_bop, bop_index_in_window, removed, False
    return to_be_padded_bop, bop_index_in_window, removed, True

def use_value_for_key(target_value, map):
    for key, value in map.items():
        if value == target_value:
            return key
def tokenizeAndGetIndex(text, processed_dict, tokenized_new_text, max_num_bop):
    tokenized_index_dict = {}
    all_bop_index = []
    for index, i in enumerate(tokenized_new_text):
        # special tokens added by us
        if i >= 50265:
            if i not in tokenized_index_dict.keys():
                tokenized_index_dict[i] = []
            tokenized_index_dict[i] = tokenized_index_dict[i] + [index]
        # if <bopi> 
        if i >= 50265 and i % 2 == 1:     
            all_bop_index.append(index)

    # create bop_to_eop and bop_to_label
    bop_to_eop = {}
    bop_to_label = {}
    bop_to_start_end = {}
    for i in range(0, max_num_bop + 1):
        bop_index_list = tokenized_index_dict[50265 + i*2]
        eop_index_list = tokenized_index_dict[50266 + i*2]
        assert len(bop_index_list) == len(eop_index_list)
        
        processed_row = processed_dict[i]
        assert len(bop_index_list) == len(processed_row)
        # print(f"{i}bop_index_list-{bop_index_list}")
        for j in range(len(bop_index_list)):
            bop_to_start_end[bop_index_list[j]] = (processed_row[j]['start'], processed_row[j]['end'])
            bop_to_label[bop_index_list[j]] = processed_row[j]['one_hot']
            bop_to_eop[bop_index_list[j]] = eop_index_list[j]
    assert len(all_bop_index) == len(bop_to_eop)
    assert len(all_bop_index) == len(bop_to_label)
    return all_bop_index, bop_to_eop, bop_to_label, bop_to_start_end
def insertBopAndEop(text, article, start_token, end_token, shift_in_char_index, debug, original_text):
    not_processed = []
    processed = []
    curr_end = 0
    added_char = 0
    token_length = len(start_token)
    for index,row in enumerate(article):
        if row['start'] < curr_end: 
            not_processed.append(row)
            shift_in_start, shift_int_end = shift_in_char_index.get((row['start'],row['end']), (0, 0))
            if curr_end < row['end']:
                shift_in_char_index[(row['start'],row['end'])] = (shift_in_start + added_char - token_length, shift_int_end + added_char)
            else:
                shift_in_char_index[(row['start'],row['end'])] = (shift_in_start + added_char - token_length, shift_int_end + added_char - token_length)
        else:  
            shift_in_start, shift_int_end = shift_in_char_index.get((row['start'],row['end']), (0, 0))
            text = insert(text, row['start'] + shift_in_start + added_char, start_token)
            added_char += token_length
            text = insert(text, row['end'] + shift_int_end + added_char, end_token)
            added_char += token_length
            curr_end = row['end'] 
            processed.append(row)
            if debug:
                print(original_text[row['start']: row['end']])
                print(shift_in_start)
                print(shift_int_end)
                print("||")
                print(text)
                print("-------")
                f = open(f"{row['start']}: {row['end']}.txt", "w")
                f.write(text)
                f.close()
    return not_processed, shift_in_char_index, text, processed
def insert(text, insert_position, insert_string):
    return text[:insert_position] + insert_string + text[insert_position:]

if split: df = pickle.load( open( f"../split_data/processed_data/{curr_split}/{train_or_dev}.pkl", "rb" ) )
else: df = pickle.load( open( f"../processed_data/{train_or_dev}.pkl", "rb" ) )
df = df.astype('object')
df = df.groupby(['article_id','start','end'])['techniques'].apply(list).reset_index()	
df['one_hot'] = df.apply(lambda x: [1 if i in x['techniques'] else 0 for i in range(14)], axis=1)
df.sort_values(by=['end'], ascending=False, inplace=True)
df.sort_values(by=['article_id', 'start'], inplace=True)

# find the list of articles that we need  
article_set = set(df['article_id'])
number_processed_articles = 0
max_num_bop = 0
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", additional_special_tokens=("<bop0>", "<eop0>", "<bop1>", "<eop1>", "<bop2>", "<eop2>", "<bop3>", "<eop3>"))
new_df = pd.DataFrame(columns = ['article_id', 'window_start_index', 'bop_index','eop_index', 'context_len_for_prediction', 'start_in_context', 'end_in_context', 'labels', 'labels_mask', f'labels_in_{context_length}', f'{context_length}_chunk_tokens', 'attention_mask', 'min_two_sides'])
bop_to_start_end_all = {}
for article_id in article_set:
    article = df.loc[df['article_id'] == article_id]
    curr_max_num_bop, new_df, bop_to_start_end, bop_processed = processArticle(article_id, article, df, tokenizer, new_df)
    not_processed = set(bop_to_start_end.keys()) - bop_processed
    assert len(not_processed) == 0
    # for i in not_processed:
    #     print(f"{article_id} - {bop_to_start_end[i]}")
    bop_to_start_end_all[article_id] = bop_to_start_end
    max_num_bop = max(curr_max_num_bop, max_num_bop)
    number_processed_articles += 1
print(f"final max:{max_num_bop}")
# sanity check that all the bops are found
print("Successfully processed", number_processed_articles)

if split: 
    pickle.dump(new_df, open(f'../split_data/processed_data/{curr_split}/{train_or_dev}_{context_length}_chunk_long.pkl', "wb"))
    new_df.to_csv(f'../split_data/processed_data/{curr_split}/{train_or_dev}_{context_length}_chunk.csv', index = False)
else:
    pickle.dump(new_df, open(f'../processed_data/{train_or_dev}_{context_length}_chunk_long.pkl', "wb"))	
    new_df.to_csv(f'../processed_data/{train_or_dev}_{context_length}_chunk.csv', index = False)


# longest one: 999001621
# [776345502]
# 3: 766632016, 703056647(difficult)

print(total_windows)
print(truncated_windows)

# article_id	
# window_start_index: start index of the window	
# bop_index: local index of the bops 	
# eop_index: local index of the eops 	
# context_len_for_prediction	
# start_in_context,end_in_context: needed in prediction, to map back to the correct start and end 	
# labels: labels in multiple hots 	
# labels_mask: a vector of 256, 1 when there is a label	
# labels_in_256: 256 * 14 	
# 256_chunk_tokens: tokens 	
# attention_mask: mask out those padded when near the end of an article 	
# min_two_sides: min of the two sides in context	

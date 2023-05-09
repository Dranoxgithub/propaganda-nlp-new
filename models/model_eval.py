from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import pickle
import os
import evaluation.scorer as scorer
from sklearn.metrics import f1_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict_and_write_to_file_for_span(model, loader, output_file_name):
    model.eval()
    total_loss = 0

    class_frequency = [2448, 1241, 766, 534, 559, 338, 316, 227, 169, 158, 129, 93, 136, 77]
    # [2123, 1058, 621, 466, 493, 294, 229, 209, 129, 144, 107, 76, 107, 72]
    max_freq = max(class_frequency)
    pos_weight = torch.tensor([max_freq/i for i in class_frequency]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    num_batches = 0
    with torch.no_grad() and open(output_file_name, "w") as fout:
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bop_index = batch['bop_index'].to(device)
            start = batch['start']
            end = batch['end']
            article_id = batch['article_id']
            output = model(input_ids, attention_mask, bop_index)

            total_loss += criterion(output, batch['label'].float().to(device)).item()
            num_batches += 1 

            num_predictions_list = torch.sum(batch['label'], dim=1).tolist()
            max_num_predictions = max(num_predictions_list)
            max_prob, max_index = torch.topk(output, max_num_predictions, dim = 1)
            for i in range(len(num_predictions_list)):
                for j in range(num_predictions_list[i]):
                    fout.write(f"{article_id[i].item()}\t{max_index[i][j]}\t{start[i].item()}\t{end[i].item()}\n")  
    return total_loss / num_batches
def predict_and_write_to_file_for_span_CE(model, loader, output_file_name):
    model.eval()
    total_loss = 0

    class_frequency = [2448, 1241, 766, 534, 559, 338, 316, 227, 169, 158, 129, 93, 136, 77]
    # [2123, 1058, 621, 466, 493, 294, 229, 209, 129, 144, 107, 76, 107, 72]
    max_freq = max(class_frequency)
    pos_weight = torch.tensor([max_freq/i for i in class_frequency]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    num_batches = 0
    with torch.no_grad() and open(output_file_name, "w") as fout:
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bop_index = batch['bop_index'].to(device)
            start = batch['start']
            end = batch['end']
            article_id = batch['article_id']
            output = model(input_ids, attention_mask, bop_index)

            total_loss += criterion(output, batch['labels_CE'].to(device)).item()
            num_batches += 1 

            num_predictions_list = torch.sum(batch['label'], dim=1).tolist()
            max_num_predictions = max(num_predictions_list)
            max_prob, max_index = torch.topk(output, max_num_predictions, dim = 1)
            for i in range(len(num_predictions_list)):
                for j in range(num_predictions_list[i]):
                    fout.write(f"{article_id[i].item()}\t{max_index[i][j]}\t{start[i].item()}\t{end[i].item()}\n")  
    return total_loss / num_batches
def predict_and_write_to_file_for_chunk(context_length, model, loader, output_file_name):
    model.eval()
    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels_mask = batch['labels_mask'].to(device)
            article_id = batch['article_id']
            # label_one_hot = batch['labels'].float().to(device)

            output = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions
            output = torch.nn.functional.softmax(output, dim = 2)

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
                
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_value, max_index = torch.topk(output[i, j, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                
                # for j in labels_mask[i].nonzero().view(-1):
                #     j = j.item()
                #     # first know need to detect how many predictions 
                #     num_predictions = torch.sum(label_one_hot[i, j, :]).item()
                #     num_predictions = int(num_predictions)
                #     print(num_predictions)
                #     # collect the corresponding start and end we need to write
                #     start = start_index_list[j]
                #     end = end_index_list[j]

                #     curr_context_len = context_len_for_prediction[j]
                #     curr_min_two_sides = min_two_sides[j]
                #     max_value, max_index = torch.topk(output[i, j, :], num_predictions)
                #     if (start, end) not in predicting.keys():
                #         predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                #     elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                #         predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
        write_article(curr_article_id, predicting, fout)
def predict_and_write_to_file_for_chunk_long(context_length, model, loader, output_file_name):
    model.eval()
    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels_mask = batch['labels_mask'].to(device)
            article_id = batch['article_id']
            # label_one_hot = batch['labels'].float().to(device)
            global_attention_mask = batch['global_attention_mask'].to(device) # expect batch_size * seq_length

            output = model(input_ids, attention_mask, global_attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions
            output = torch.nn.functional.softmax(output, dim = 2)

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
                
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_value, max_index = torch.topk(output[i, j, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
        write_article(curr_article_id, predicting, fout)
def predict_and_write_to_file_for_chunk_crf(context_length, model, loader, output_file_name):
    def backtrace(bps, last_labels):
        """
        Obtain the highest scoring label sequence by tracing backward with backpointers.

        args
        - bps: batchsize x M x K
        - last_labels: batchsize tensor of indices of highest scoring label @ final timestep
        
        returns
        - preds: a batchsize x M tensor containing predicted label indices
        """
        bsz, M = bps.shape[0], bps.shape[1]
        preds = torch.zeros(bps.shape[0], M, dtype=int)
        preds[:, - 1] = last_labels
        for m in reversed(range(1, M)):
            preds[:, m-1] = bps[np.arange(bsz), m, preds[:, m]]
        return preds

    def viterbi(psi0, psis):
        """
        Computes the Viterbi algorithm.

        args
        - psi0: a tensor of real scores (which can be < 0), of dimension batchsize x K
        - psis: a tensor of real scores (which can be < 0), of dimension batchsize x M-1 x K x K

        returns
        - max_scores: a batchsize tensor of max scores for each sequence
        - preds: a batchsize x M tensor of label indices, where each index is between 0 and K-1
        """
        bsz, K = psi0.size() # batchsize, nlabels
        M = psis.size(1) + 1 # sequence length

        # make our viterbi table data structures
        vtable = psis.new(bsz, M, K).zero_() # stores max scores
        bps = torch.zeros(bsz, M, K, dtype=torch.long, device=vtable.device) # stores backpointers

        # base case: max score of any sequence of length 1 ending in label k is just
        # the score of the first position having label k.
        vtable[:, 0].copy_(psi0)

        # now we fill out the rest of the table...

        ### BEGIN YOUR CODE HERE!!!!
        # your code should probably begin with: for m in range(1, M):
        for m in range(1, M):
            prev_v = vtable[:, m-1] # expect batchsize * K
            prev_v = prev_v.unsqueeze(2) #expect batchsize * K * 1
            prev_v = prev_v.expand(-1, -1, K) # expect batchsize * K * K
            currTable = psis[:, m - 1, :, :] # expect batchsize x K x K
            currTable = currTable + prev_v
            (max, index) = torch.max(currTable, 1) # expect batchsize * K
            vtable[:, m, :] = max
            bps[:, m, :] = index 
        last_row_v = vtable[:, -1, :] # expect batchsize * K
        (max_scores, last_labels) = torch.max(last_row_v, 1)
        ### END YOUR CODE HERE

        # Obtain highest scoring label sequence by tracing backward thru bps below. 
        # last_labels is a batchsize-length tensor containing the indices of 
        # the highest scoring label @ final timestep (for each sequence in the batch).
        # It should be obtained in your code block above.
        argmax_seqs = backtrace(bps, last_labels)    
        return max_scores, argmax_seqs
    def get_more_predictions(best_score_seq, k, psi0, psis, num_predictions):
        # - psi0, psis corresponding to this batch element: 
        # psi0: num_labels; psis: (label_length - 1) x num_labels x num_labels
        
        # pick the next highest score
        predictions = []
        predictions.append(best_score_seq[k])
        if k == 0: 
            test = psi0
        else:
            test = psis[k - 1, best_score_seq[k - 1], :] # expect num_labels
        max_prob, max_index = torch.topk(test, 14, dim = 0)
        for i in max_index:
            if i not in predictions:
                predictions.append(i)
                if len(predictions) == num_predictions:
                    return predictions
        # print(best_score_seq)
        # print(psi0.shape)
        # print(psis.shape)
        # assert False

    model.eval()
    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            article_id = batch['article_id']

            bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 1024).to(device) # expect batch_size * max_length * hidden_size
            psi0, psis = model(input_ids, attention_mask, bop_index_list) # (batch_size, sequence_length, num_labels)
            max_scores, argmax_seqs = viterbi(psi0, psis)

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
                
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_index = [argmax_seqs[i][k].item()]

                    # get more predictions 
                    if num_predictions > 1:
                        max_index = get_more_predictions(argmax_seqs[i].tolist(), k, psi0[i, :], psis[i, :, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (0, max_index, curr_context_len, curr_min_two_sides) # put 0 in position 0 just so that write_article not affected
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (0, max_index, curr_context_len, curr_min_two_sides)
        write_article(curr_article_id, predicting, fout)
def write_article(article_id, predicting, fout):
    for key, value in predicting.items():
        predictions = value[1]
        for prediction in predictions:
            fout.write(f"{article_id}\t{int(prediction)}\t{int(key[0])}\t{int(key[1])}\n")
    
def score(user_submission_file, gold_file, propaganda_techniques_list_file="evaluation/propaganda-techniques-names-semeval2020task11.txt"):
    sort_before_matching(user_submission_file)
    # sort_before_matching(gold_file)
    return scorer.main(user_submission_file, gold_file, None, propaganda_techniques_list_file)
    
def sort_before_matching(user_submission_file):
    df = pd.read_csv(user_submission_file, sep = "\t", header=None, names=['article_id', 'techniques', "start", "end"])
    df.sort_values(by=['end'], ascending=False, inplace=True)
    df.sort_values(by=['article_id', 'start'], inplace=True)
    df.to_csv(user_submission_file, header=None, index=None, sep='\t', mode='w')

def write_best_f1_to_file(best_f1, file_name):
    with open(file_name, "w") as fout:
        fout.write(f"best_f1: {best_f1}")
def predict_and_write_to_file_for_chunk_transformer(context_length, model, loader, output_file_name):
    model.eval()
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            article_id = batch['article_id']
            # transformer on top 
            bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 1024).to(device) # expect batch_size * max_length * hidden_size
            output = model(input_ids, attention_mask, bop_index_list) # (batch_size, max_length, num_labels)

            # normalize so that it is comparable across different predictions
            output = torch.nn.functional.softmax(output, dim = 2)

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
                
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_value, max_index = torch.topk(output[i, k, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
        write_article(curr_article_id, predicting, fout)

def predict_and_write_to_file_for_chunk_hier(context_length, model, loader, output_file_name, aux_weight):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret

    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        num_batches = 0
        total_loss = 0
        for batch in loader:
            num_batches += 1
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            article_id = batch['article_id']

            output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions

            # compute auxiliary prediction
            bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
            for i in range(7):
                test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
            gathered_output = torch.gather(output, 1, bop_index_list)

            # loss  
            loss_in_full_dim = criterion(torch.transpose(gathered_output, 1, 2), batch["labels_new_CE"].to(device)) # expect batch_size * max_length
            new_label_mask = batch['new_label_mask'].to(device)
            loss_in_full_dim = loss_in_full_dim * new_label_mask
            ori_loss = torch.sum(loss_in_full_dim) / torch.sum(new_label_mask)  
            total_loss += ori_loss.item()
            # normalize
            for i in range(7):
                test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
            gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)

            # compute auxiliary new
            aux_prob = compute_auxiliary_preds(test) 
            combi_prob = aux_prob * aux_weight + gathered_output * (1 - aux_weight) # expect max_length * 14

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
            
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
  
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_value, max_index = torch.topk(combi_prob[i, k, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                
        write_article(curr_article_id, predicting, fout)
        return total_loss / num_batches
def calculate_entropy(model, loader):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret
    model.eval()
    num_active_classifers = {}
    entropy_sum = {}
    for i in range(7):
        num_active_classifers[i] = 0
        entropy_sum[i] = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        article_id = batch['article_id']

        output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)

        # compute auxiliary prediction
        bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
        for i in range(7):
            test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
        gathered_output = torch.gather(output, 1, bop_index_list)

        # normalize
        for i in range(7):
            test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
        gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)


        for classifier_num in range(7):
            mask = batch[f'classifier_mask_{classifier_num}']
            num_active_classifers[classifier_num] += torch.sum(mask).item()

            mask_expand = mask.unsqueeze(-1).repeat(1, 1, classifier_num_label[classifier_num]).to(device)
            prob_full_dim = mask_expand * test[classifier_num] # batch size * max length * number of labels for classifier
            entropy_full_dim = torch.special.entr(prob_full_dim)
            entropy_sum[classifier_num] += torch.sum(entropy_full_dim).item()

    average_entropy = {}
    for classifier_num in range(7):
        print(classifier_num)
        print(entropy_sum[classifier_num])
        print(num_active_classifers[classifier_num])
        average_entropy[classifier_num] = entropy_sum[classifier_num] / num_active_classifers[classifier_num]
    return average_entropy
def calculate_classifier_accuracy(model, loader):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret
    model.eval()
    num_active_classifers = {}
    correct = {}
    for i in range(7):
        num_active_classifers[i] = 0
        correct[i] = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        article_id = batch['article_id']

        output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)

        # compute auxiliary prediction
        bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
        for i in range(7):
            test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
        gathered_output = torch.gather(output, 1, bop_index_list)

        # normalize
        for i in range(7):
            test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
        gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)


        for classifier_num in range(7):
            mask = batch[f'classifier_mask_{classifier_num}'].to(device)
            num_active_classifers[classifier_num] += torch.sum(mask).item()

            gold_index = batch[f'classifier_{classifier_num}'].to(device) # batch_size * max_length
            _, pred_index = torch.topk(test[classifier_num], 1, dim=2)
            pred_index = pred_index.squeeze(-1)

            correct_index = (gold_index == pred_index) * 1
            correct_after_mask = mask * correct_index # batch size * max length
            
            correct[classifier_num] += torch.sum(correct_after_mask).item()

    accuracy = {}
    for classifier_num in range(7):
        print(classifier_num)
        print("correct")
        print(correct[classifier_num])
        print(num_active_classifers[classifier_num])
        accuracy[classifier_num] = correct[classifier_num] / num_active_classifers[classifier_num]
    return accuracy
def calculate_entropy_for_inactive(model, loader):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret
    model.eval()
    num_active_classifers = {}
    entropy_sum = {}
    for i in range(7):
        num_active_classifers[i] = 0
        entropy_sum[i] = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        article_id = batch['article_id']

        output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)

        # compute auxiliary prediction
        bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
        for i in range(7):
            test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
        gathered_output = torch.gather(output, 1, bop_index_list)

        # normalize
        for i in range(7):
            test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
        gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)

        padding_mask = batch[f'classifier_mask_0'] # since classifier 0 is always in use 
        for classifier_num in range(7):
            mask = batch[f'classifier_mask_{classifier_num}']
            inactivie_mask = padding_mask - mask
            num_active_classifers[classifier_num] += torch.sum(inactivie_mask).item()

            mask_expand = inactivie_mask.unsqueeze(-1).repeat(1, 1, classifier_num_label[classifier_num]).to(device)
            prob_full_dim = mask_expand * test[classifier_num] # batch size * max length * number of labels for classifier
            entropy_full_dim = torch.special.entr(prob_full_dim)
            entropy_sum[classifier_num] += torch.sum(entropy_full_dim).item()

    average_entropy = {}
    # ignore classifier 0 since it is always in use
    for classifier_num in range(1, 7):
        print(classifier_num)
        print(entropy_sum[classifier_num])
        print(num_active_classifers[classifier_num])
        average_entropy[classifier_num] = entropy_sum[classifier_num] / num_active_classifers[classifier_num]
    return average_entropy   
# this assesses the f1 of the aux classifier
# for every classifier, looking at the datapoints where this classifier should be active
# the prediction is just the label with higher probability predicted by this classifier
def calculate_classifier_f1(model, loader):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret
    model.eval()
    num_active_classifers = {}
    correct = {}
    pred = {}
    gold = {}
    for i in range(7):
        num_active_classifers[i] = 0
        correct[i] = 0
        pred[i] = []
        gold[i] = []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        article_id = batch['article_id']
        
        output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)

        # compute auxiliary prediction
        bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
        for i in range(7):
            test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
        gathered_output = torch.gather(output, 1, bop_index_list)

        # normalize
        for i in range(7):
            test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
        gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)

        # we are only evaluating the aux classifiers, we ignore the gathered otuput from the orginal classifier

        for classifier_num in range(7):
            mask = batch[f'classifier_mask_{classifier_num}'].to(device) # batch_size * max_length
            num_active_classifers[classifier_num] += torch.sum(mask).item()

            gold_index = batch[f'classifier_{classifier_num}'].to(device) # batch_size * max_length
            _, pred_index = torch.topk(test[classifier_num], 1, dim=2)
            pred_index = pred_index.squeeze(-1)

            batch_size = gold_index.shape[0]
            max_length = gold_index.shape[1]

            for i in range(batch_size):
                # loop through where there is 1 in the mask, meaning active for this classifier
                for j in range(max_length):
                    if mask[i, j] == 1:
                        pred[classifier_num].append(pred_index[i, j].item())
                        gold[classifier_num].append(gold_index[i, j].item())
    f1 = {}
    for classifier_num in range(7):
        if classifier_num == 0: 
            print(gold[classifier_num])
            print("pred")
            print(pred[classifier_num])
        f1[classifier_num] =  f1_score(gold[classifier_num], pred[classifier_num], average="macro")
    return f1

def predict_and_write_to_file_for_chunk_CUDA(context_length, model, loader, output_file_name):
    model.eval()
    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels_mask = batch['labels_mask'].to(device)
            article_id = batch['article_id']
            # label_one_hot = batch['labels'].float().to(device)

            output = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions
            output = torch.nn.functional.softmax(output, dim = 2)

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
                
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_value, max_index = torch.topk(output[i, j, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
        write_article(curr_article_id, predicting, fout)
    
def predict_and_write_to_file_for_chunk_hier_early_stopping(context_length, model, loader, output_file_name, aux_weight, threshold):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret
    def calculate_for_every_node(prob):    
        result = {}
        result['classifier2'] = prob[7] + prob[9] + prob[10] + prob[11]
        result['classifier1'] = result['classifier2'] + prob[12]

        result['classifier5'] = prob[2] + prob[4] + prob[5] + prob[8]
        result['classifier6'] = prob[0] + prob[3] + prob[6]
        result['classifier4'] = result['classifier5'] + result['classifier6']
        result['classifier3'] = prob[1] + result['classifier4'] + prob[13]

        result['classifier0'] = result['classifier1'] + result['classifier3']
        return result
    def process(prob_result, index_candidate, threshold, probability):
        # parents_to_kids = {'classifier0': ['classifier1', 'classifier3'], 'classifier1': [12, 'classifier2'], 'classifier3': [1, 'classifier4', 13], 12: [], 'classifier2': [7, 9, 10, 11], 1: [], 'classifier4': ['classifier5', 'classifier6'], 13: [], 7: [], 9: [], 10: [], 11: [], 'classifier5': [2, 4, 5, 8], 'classifier6': [0, 3, 6], 2: [], 4: [], 5: [], 8: [], 0: [], 3: [], 6: []}
        kids_to_parents = {'classifier1': 'classifier0', 'classifier3': 'classifier0', 12: 'classifier1', 'classifier2': 'classifier1', 1: 'classifier3', 'classifier4': 'classifier3', 13: 'classifier3', 7: 'classifier2', 9: 'classifier2', 10: 'classifier2', 11: 'classifier2', 'classifier5': 'classifier4', 'classifier6': 'classifier4', 2: 'classifier5', 4: 'classifier5', 5: 'classifier5', 8: 'classifier5', 0: 'classifier6', 3: 'classifier6', 6: 'classifier6'}
        
        curr_candidate = kids_to_parents[index_candidate]
        while curr_candidate != 'classifier0':
            if prob_result[curr_candidate] >= threshold:
                return prob_result[curr_candidate], curr_candidate
            else:
                curr_candidate = kids_to_parents[curr_candidate]
        curr_candidate = 'classifier0'
        return prob_result[curr_candidate], curr_candidate
    def early_stopping(probability, num_predictions, threshold):
        max_value_candidates, max_index_candidates = torch.topk(probability, num_predictions)
        max_value = []
        max_index = []
        prob_result = calculate_for_every_node(probability.tolist())
        for i in range(len(max_value_candidates)):
            if max_value_candidates[i] >= threshold:
                max_value.append(max_value_candidates[i].item())
                max_index.append(max_index_candidates[i].item())
            else:
                curr_value, curr_index = process(prob_result, max_index_candidates[i].item(), threshold, probability.tolist())
                max_value.append(curr_value)
                max_index.append(curr_index)
        return max_value, max_index
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        num_batches = 0
        total_loss = 0
        for batch in loader:
            num_batches += 1
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            article_id = batch['article_id']

            output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions

            # compute auxiliary prediction
            bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
            for i in range(7):
                test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
            gathered_output = torch.gather(output, 1, bop_index_list)

            # loss  
            loss_in_full_dim = criterion(torch.transpose(gathered_output, 1, 2), batch["labels_new_CE"].to(device)) # expect batch_size * max_length
            new_label_mask = batch['new_label_mask'].to(device)
            loss_in_full_dim = loss_in_full_dim * new_label_mask
            ori_loss = torch.sum(loss_in_full_dim) / torch.sum(new_label_mask)  
            total_loss += ori_loss.item()
            # normalize
            for i in range(7):
                test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
            gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)

            # compute auxiliary new
            aux_prob = compute_auxiliary_preds(test) 
            combi_prob = aux_prob * aux_weight + gathered_output * (1 - aux_weight) # expect max_length * 14

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article_early_stopping(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
            
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_value, max_index = early_stopping(combi_prob[i, k, :].detach(), num_predictions, threshold)
                    # max_value, max_index = torch.topk(combi_prob[i, k, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value, max_index, curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value, max_index, curr_context_len, curr_min_two_sides)
                
        write_article_early_stopping(curr_article_id, predicting, fout)
        return total_loss / num_batches
# def write_article_early_stopping(article_id, predicting, fout):
#     for key, value in predicting.items():
#         predictions = value[1]
#         for prediction in predictions:
#             fout.write(f"{article_id}\t{prediction}\t{int(key[0])}\t{int(key[1])}\n")
def write_article_early_stopping_with_probability(article_id, predicting, file):
    with open(file, 'a') as fout:
        for key, value in predicting.items():
            probability = value[0]
            predictions = value[1]
            for i in range(len(predictions)):
                test = '{0:.7g}'.format(probability[i])
                fout.write(f"{article_id} {predictions[i]} {int(key[0])} {int(key[1])} {probability[i]}\n")
                print(f"{article_id}\t{predictions[i]}\t{int(key[0])}\t{int(key[1])}\t{test}")
def predict_and_write_to_file_for_chunk_early_stopping(context_length, model, loader, output_file_name, threshold):
    model.eval()
    def calculate_for_every_node(prob):    
        result = {}
        result['classifier2'] = prob[7] + prob[9] + prob[10] + prob[11]
        result['classifier1'] = result['classifier2'] + prob[12]

        result['classifier5'] = prob[2] + prob[4] + prob[5] + prob[8]
        result['classifier6'] = prob[0] + prob[3] + prob[6]
        result['classifier4'] = result['classifier5'] + result['classifier6']
        result['classifier3'] = prob[1] + result['classifier4'] + prob[13]

        result['classifier0'] = result['classifier1'] + result['classifier3']
        return result
    def process(prob_result, index_candidate, threshold, probability):
        # parents_to_kids = {'classifier0': ['classifier1', 'classifier3'], 'classifier1': [12, 'classifier2'], 'classifier3': [1, 'classifier4', 13], 12: [], 'classifier2': [7, 9, 10, 11], 1: [], 'classifier4': ['classifier5', 'classifier6'], 13: [], 7: [], 9: [], 10: [], 11: [], 'classifier5': [2, 4, 5, 8], 'classifier6': [0, 3, 6], 2: [], 4: [], 5: [], 8: [], 0: [], 3: [], 6: []}
        kids_to_parents = {'classifier1': 'classifier0', 'classifier3': 'classifier0', 12: 'classifier1', 'classifier2': 'classifier1', 1: 'classifier3', 'classifier4': 'classifier3', 13: 'classifier3', 7: 'classifier2', 9: 'classifier2', 10: 'classifier2', 11: 'classifier2', 'classifier5': 'classifier4', 'classifier6': 'classifier4', 2: 'classifier5', 4: 'classifier5', 5: 'classifier5', 8: 'classifier5', 0: 'classifier6', 3: 'classifier6', 6: 'classifier6'}
        
        curr_candidate = kids_to_parents[index_candidate]
        while curr_candidate != 'classifier0':
            if prob_result[curr_candidate] >= threshold:
                return prob_result[curr_candidate], curr_candidate
            else:
                curr_candidate = kids_to_parents[curr_candidate]
        curr_candidate = 'classifier0'
        return prob_result[curr_candidate], curr_candidate
    def early_stopping(probability, num_predictions, threshold):
        max_value_candidates, max_index_candidates = torch.topk(probability, num_predictions)
        max_value = []
        max_index = []
        prob_result = calculate_for_every_node(probability.tolist())
        for i in range(len(max_value_candidates)):
            if max_value_candidates[i] >= threshold:
                max_value.append(max_value_candidates[i].item())
                max_index.append(max_index_candidates[i].item())
            else:
                curr_value, curr_index = process(prob_result, max_index_candidates[i].item(), threshold, probability.tolist())
                max_value.append(curr_value)
                max_index.append(curr_index)
        return max_value, max_index
    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad() and open(output_file_name, "w") as fout:
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels_mask = batch['labels_mask'].to(device)
            article_id = batch['article_id']
            # label_one_hot = batch['labels'].float().to(device)

            output = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions
            output = torch.nn.functional.softmax(output, dim = 2)

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article_early_stopping(curr_article_id, predicting, fout)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
                
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    # max_value, max_index = torch.topk(output[i, j, :], num_predictions)
                    max_value, max_index = early_stopping(output[i, j, :].detach(), num_predictions, threshold)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value, max_index, curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value, max_index, curr_context_len, curr_min_two_sides)
        write_article_early_stopping(curr_article_id, predicting, fout)






def predict_and_write_to_file_for_chunk_hier_early_stopping_1(context_length, model, loader, output_file_name, aux_weight, threshold):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret
    def calculate_for_every_node(prob):    
        result = {}
        result['classifier2'] = prob[7] + prob[9] + prob[10] + prob[11]
        result['classifier1'] = result['classifier2'] + prob[12]

        result['classifier5'] = prob[2] + prob[4] + prob[5] + prob[8]
        result['classifier6'] = prob[0] + prob[3] + prob[6]
        result['classifier4'] = result['classifier5'] + result['classifier6']
        result['classifier3'] = prob[1] + result['classifier4'] + prob[13]

        result['classifier0'] = result['classifier1'] + result['classifier3']
        return result
    def process(prob_result, index_candidate, threshold, probability):
        # parents_to_kids = {'classifier0': ['classifier1', 'classifier3'], 'classifier1': [12, 'classifier2'], 'classifier3': [1, 'classifier4', 13], 12: [], 'classifier2': [7, 9, 10, 11], 1: [], 'classifier4': ['classifier5', 'classifier6'], 13: [], 7: [], 9: [], 10: [], 11: [], 'classifier5': [2, 4, 5, 8], 'classifier6': [0, 3, 6], 2: [], 4: [], 5: [], 8: [], 0: [], 3: [], 6: []}
        kids_to_parents = {'classifier1': 'classifier0', 'classifier3': 'classifier0', 12: 'classifier1', 'classifier2': 'classifier1', 1: 'classifier3', 'classifier4': 'classifier3', 13: 'classifier3', 7: 'classifier2', 9: 'classifier2', 10: 'classifier2', 11: 'classifier2', 'classifier5': 'classifier4', 'classifier6': 'classifier4', 2: 'classifier5', 4: 'classifier5', 5: 'classifier5', 8: 'classifier5', 0: 'classifier6', 3: 'classifier6', 6: 'classifier6'}
        
        curr_candidate = kids_to_parents[index_candidate]
        while curr_candidate != 'classifier0':
            if prob_result[curr_candidate] >= threshold:
                return prob_result[curr_candidate], curr_candidate
            else:
                curr_candidate = kids_to_parents[curr_candidate]
        curr_candidate = 'classifier0'
        return prob_result[curr_candidate], curr_candidate
    def early_stopping(probability, num_predictions, threshold):
        max_value_candidates, max_index_candidates = torch.topk(probability, num_predictions)
        max_value = []
        max_index = []
        prob_result = calculate_for_every_node(probability.tolist())
        for i in range(len(max_value_candidates)):
            if max_value_candidates[i] >= threshold:
                max_value.append(max_value_candidates[i].item())
                max_index.append(max_index_candidates[i].item())
            else:
                curr_value, curr_index = process(prob_result, max_index_candidates[i].item(), threshold, probability.tolist())
                # max_value.append(curr_value)
                max_value.append(max_value_candidates[i].item())
                max_index.append(curr_index)
        return max_value, max_index
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad():
        predicting = {} # index: (prob, label)
        curr_article_id = 0
        num_batches = 0
        total_loss = 0
        for batch in loader:
            num_batches += 1
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            article_id = batch['article_id']

            output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions

            # compute auxiliary prediction
            bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
            for i in range(7):
                test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
            gathered_output = torch.gather(output, 1, bop_index_list)

            # loss  
            loss_in_full_dim = criterion(torch.transpose(gathered_output, 1, 2), batch["labels_new_CE"].to(device)) # expect batch_size * max_length
            new_label_mask = batch['new_label_mask'].to(device)
            loss_in_full_dim = loss_in_full_dim * new_label_mask
            ori_loss = torch.sum(loss_in_full_dim) / torch.sum(new_label_mask)  
            total_loss += ori_loss.item()
            # normalize
            for i in range(7):
                test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
            gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)

            # compute auxiliary new
            aux_prob = compute_auxiliary_preds(test) 
            combi_prob = aux_prob * aux_weight + gathered_output * (1 - aux_weight) # expect max_length * 14

            for i in range(len(article_id)): # loop through every batch element 
                start_index_list = batch['start_in_context'][i].tolist()
                end_index_list = batch['end_in_context'][i].tolist()
                context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                min_two_sides = batch['min_two_sides'][i].tolist()
                if curr_article_id != article_id[i].item() and curr_article_id != 0:
                    write_article_early_stopping_with_probability(curr_article_id, predicting, output_file_name)
                    predicting = {}
                    curr_article_id = article_id[i].item()
                if curr_article_id == 0:
                    curr_article_id = article_id[i].item()

                window_start_index = batch['window_start_index'][i].item()
            
                # loop through indexes where there should be predictions 
                for k in range(torch.sum(batch["new_label_mask"][i])):
                    j = batch['bop_index'][i][k].item()
                    # first know need to detect how many predictions 
                    num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                    # collect the corresponding start and end we need to write
                    start = start_index_list[j]
                    end = end_index_list[j]

                    curr_context_len = context_len_for_prediction[j]
                    curr_min_two_sides = min_two_sides[j]
                    max_value, max_index = early_stopping(combi_prob[i, k, :].detach(), num_predictions, threshold)
                    
                    # if curr_article_id == 71 and int(start) == 16919:
                    #     print(end)
                    #     print(combi_prob[i, k, :])
                    #     print(max_index)
                    #     print(max_value)
                    #     assert False
                    # max_value, max_index = torch.topk(combi_prob[i, k, :], num_predictions)
                    if (start, end) not in predicting.keys():
                        predicting[(start, end)] = (max_value, max_index, curr_context_len, curr_min_two_sides)
                    elif curr_context_len > predicting[(start, end)][2] or (curr_context_len == predicting[(start, end)][2] and curr_min_two_sides > predicting[(start, end)][3]):
                        predicting[(start, end)] = (max_value, max_index, curr_context_len, curr_min_two_sides)
                
        write_article_early_stopping_with_probability(curr_article_id, predicting, output_file_name)
        return total_loss / num_batches
def sort_before_matching_1(user_submission_file):
    df = pd.read_csv(user_submission_file, sep = " ", header=None, names=['article_id', 'techniques', "start", "end", "prob"])
    df.sort_values(by=['end'], ascending=False, inplace=True)
    df.sort_values(by=['article_id', 'start'], inplace=True)
    df.to_csv(user_submission_file, header=None, index=None, sep='\t', mode='w')
def predict_and_write_to_file_for_chunk_hier_list(context_length, model, loader, output_file_name, aux_weight_list):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, max_length, _ = test[0].shape
        ret = torch.zeros(batch_size, max_length, 14).to(device)
        ret[:, :, 12] = test[0][:, :, 0] * test[1][:, :, 0] # expect max_length
        temp = test[0][:, :, 0] * test[1][:, :, 1]
        ret[:, :, 7] = temp * test[2][:, :, 0]
        ret[:, :, 9] = temp * test[2][:, :, 1]
        ret[:, :, 10] = temp * test[2][:, :, 2]
        ret[:, :, 11] = temp * test[2][:, :, 3]

        ret[:, :, 1] = test[0][:, :, 1] * test[3][:, :, 0]
        ret[:, :, 13] = test[0][:, :, 1] * test[3][:, :, 2]
        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 0]
        ret[:, :, 2] = temp * test[5][:, :, 0]
        ret[:, :, 4] = temp * test[5][:, :, 1]
        ret[:, :, 5] = temp * test[5][:, :, 2]
        ret[:, :, 8] = temp * test[5][:, :, 3]

        temp =  test[0][:, :, 1] * test[3][:, :, 1] * test[4][:, :, 1]
        ret[:, :, 0] = temp * test[6][:, :, 0]
        ret[:, :, 3] = temp * test[6][:, :, 1]
        ret[:, :, 6] = temp * test[6][:, :, 2] 
        return ret
    model.eval()
    # clear the file content for append
    for aux_weight in aux_weight_list:  
        open(output_file_name + str(aux_weight), 'w').close()
        
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # 'context_len_for_prediction', 'start_in_context', 'end_in_context'
    # bop_to_start_end = pickle.load(open(f'processed_data/dev_{context_length}_chunk_bop_to_start_end.pkl', 'rb'))
    with torch.no_grad():
        predicting = {}
        for i in aux_weight_list:
            predicting[i] = {}
        # predicting = {} # index: (prob, label)
        curr_article_id_list = [0] * len(aux_weight_list)
        # curr_article_id = 0
        num_batches = 0
        total_loss = 0
        for batch in loader:
            num_batches += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            article_id = batch['article_id']

            output, test = model(input_ids, attention_mask) # (batch_size, sequence_length, num_labels)
            # normalize so that it is comparable across different predictions

            # compute auxiliary prediction
            bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) 
            for i in range(7):
                test[i] = torch.gather(test[i], 1, bop_index_list[:, :, :classifier_num_label[i]]) 
            gathered_output = torch.gather(output, 1, bop_index_list)

            # loss  
            loss_in_full_dim = criterion(torch.transpose(gathered_output, 1, 2), batch["labels_new_CE"].to(device)) # expect batch_size * max_length
            new_label_mask = batch['new_label_mask'].to(device)
            loss_in_full_dim = loss_in_full_dim * new_label_mask
            ori_loss = torch.sum(loss_in_full_dim) / torch.sum(new_label_mask)  
            total_loss += ori_loss.item()
            # normalize
            for i in range(7):
                test[i] = torch.nn.functional.softmax(test[i], dim = 2) # expect batch_size * max_length * classifier_num_label
            gathered_output = torch.nn.functional.softmax(gathered_output, dim = 2)

            # compute auxiliary new
            aux_prob = compute_auxiliary_preds(test) 
            for z in range(len(aux_weight_list)):
                aux_weight = aux_weight_list[z]
                combi_prob = aux_prob * aux_weight + gathered_output * (1 - aux_weight) # expect max_length * 14

                for i in range(len(article_id)): # loop through every batch element 
                    start_index_list = batch['start_in_context'][i].tolist()
                    end_index_list = batch['end_in_context'][i].tolist()
                    context_len_for_prediction = batch['context_len_for_prediction'][i].tolist()
                    min_two_sides = batch['min_two_sides'][i].tolist()
                    if curr_article_id_list[z] != article_id[i].item() and curr_article_id_list[z] != 0:
                        append_article(curr_article_id_list[z], predicting[aux_weight], output_file_name + str(aux_weight_list[z]))
                        predicting[aux_weight] = {}
                        curr_article_id_list[z] = article_id[i].item()
                    if curr_article_id_list[z] == 0:
                        curr_article_id_list[z] = article_id[i].item()

                    window_start_index = batch['window_start_index'][i].item()
            
                    # loop through indexes where there should be predictions 
                    for k in range(torch.sum(batch["new_label_mask"][i])):
                        j = batch['bop_index'][i][k].item()
                        # first know need to detect how many predictions 
                        num_predictions = torch.sum(batch['labels_new'][i, k, :]).item()
                    
                        # collect the corresponding start and end we need to write
                        start = start_index_list[j]
                        end = end_index_list[j]

                        curr_context_len = context_len_for_prediction[j]
                        curr_min_two_sides = min_two_sides[j]
                        max_value, max_index = torch.topk(combi_prob[i, k, :], num_predictions)
                        if (start, end) not in predicting[aux_weight].keys():
                            predicting[aux_weight][(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
                        elif curr_context_len > predicting[aux_weight][(start, end)][2] or (curr_context_len == predicting[aux_weight][(start, end)][2] and curr_min_two_sides > predicting[aux_weight][(start, end)][3]):
                            predicting[aux_weight][(start, end)] = (max_value.tolist(), max_index.tolist(), curr_context_len, curr_min_two_sides)
        for z in range(len(aux_weight_list)):
            append_article(curr_article_id_list[z], predicting[aux_weight_list[z]], output_file_name + str(aux_weight_list[z]))
        return total_loss / num_batches
def append_article(article_id, predicting, output_file_name):
    with open(output_file_name, "a") as fout:
        for key, value in predicting.items():
            predictions = value[1]
            for prediction in predictions:
                fout.write(f"{article_id}\t{int(prediction)}\t{int(key[0])}\t{int(key[1])}\n")
def predict_and_write_to_file_for_span_hier(model, loader, output_file_name, aux_weight):
    classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
    # compute auxiliary
    def compute_auxiliary_preds(test): 
        batch_size, _ = test[0].shape
        ret = torch.zeros(batch_size, 14).to(device)
        ret[:, 12] = test[0][:, 0] * test[1][:, 0] # expect max_length
        temp = test[0][:, 0] * test[1][:, 1]
        ret[:, 7] = temp * test[2][:, 0]
        ret[:, 9] = temp * test[2][:, 1]
        ret[:, 10] = temp * test[2][:, 2]
        ret[:, 11] = temp * test[2][:, 3]

        ret[:, 1] = test[0][:, 1] * test[3][:, 0]
        ret[:, 13] = test[0][:, 1] * test[3][:, 2]
        temp =  test[0][:, 1] * test[3][:, 1] * test[4][:, 0]
        ret[:, 2] = temp * test[5][:, 0]
        ret[:, 4] = temp * test[5][:, 1]
        ret[:, 5] = temp * test[5][:, 2]
        ret[:, 8] = temp * test[5][:, 3]

        temp =  test[0][:, 1] * test[3][:, 1] * test[4][:, 1]
        ret[:, 0] = temp * test[6][:, 0]
        ret[:, 3] = temp * test[6][:, 1]
        ret[:, 6] = temp * test[6][:, 2] 
        return ret
    model.eval()
    total_loss = 0

    class_frequency = [2448, 1241, 766, 534, 559, 338, 316, 227, 169, 158, 129, 93, 136, 77]
    # [2123, 1058, 621, 466, 493, 294, 229, 209, 129, 144, 107, 76, 107, 72]
    max_freq = max(class_frequency)
    pos_weight = torch.tensor([max_freq/i for i in class_frequency]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    num_batches = 0
    with torch.no_grad() and open(output_file_name, "w") as fout:
        for batch in loader:
            prediction = []
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bop_index = batch['bop_index'].to(device)
            start = batch['start']
            end = batch['end']
            article_id = batch['article_id']
            output, test = model(input_ids, attention_mask, bop_index)


            # normalize so that it is comparable across different predictions
    
            # compute auxiliary prediction
            for i in range(7):
                test[i] = test[i][:, 0, :] # batch_size * seq_length * num_predictions_for_classifier
            
            # normalize
            for i in range(7):
                test[i] = torch.nn.functional.softmax(test[i], dim = 1) # expect batch_size * classifier_num_label
            output = torch.nn.functional.softmax(output, dim = 1)

            # compute auxiliary new
            aux_prob = compute_auxiliary_preds(test)
            combi_prob = aux_prob * aux_weight + output * (1 - aux_weight) # expect max_length * 14


            total_loss += criterion(output, batch['label'].float().to(device)).item()
            num_batches += 1 

            num_predictions_list = torch.sum(batch['label'], dim=1).tolist()
            max_num_predictions = max(num_predictions_list)
            max_prob, max_index = torch.topk(combi_prob, max_num_predictions, dim = 1)
            for i in range(len(num_predictions_list)):
                for j in range(num_predictions_list[i]):
                    fout.write(f"{article_id[i].item()}\t{max_index[i][j]}\t{start[i].item()}\t{end[i].item()}\n")  
    return total_loss / num_batches

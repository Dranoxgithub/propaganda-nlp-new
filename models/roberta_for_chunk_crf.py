from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch
import model_eval
import wandb
turn_on_wandb = True
global_context_length = 0
fix_crf = False
# gold_file = "evaluation/gold/dev-task-flc-tc.labels.txt"
# gold_file = "split_data/datasets/6/dev-task-flc-tc.labels.txt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BERTClass(torch.nn.Module):
    def __init__(self, model_name, num_labels, dropout):
        super(BERTClass, self).__init__()
        self.num_labels = num_labels
        # self.project_down = torch.nn.Linear(1024, num_labels)
        configuration = RobertaConfig.from_pretrained('roberta-large', num_labels=num_labels, hidden_dropout_prob=dropout)
        self.roberta = RobertaModel.from_pretrained("roberta-large", config=configuration)
        self.roberta.resize_token_embeddings(50265 + 2 * 4)

        self.hidden_size = 1024
        self.theta_emit = torch.nn.Linear(self.hidden_size, self.num_labels)
        if fix_crf:
            self.transitions = torch.nn.Parameter(torch.empty(self.num_labels, self.num_labels))
            torch.nn.init.uniform_(self.transitions, -0.1, 0.1)
        else:
            self.theta_trans = torch.nn.Linear(self.hidden_size, self.num_labels**2)
    def forward(self, ids, mask, bop_index_list):
        batchsize = ids.shape[0]
        outputs = self.roberta(ids, mask) 
        hidden = outputs.last_hidden_state # (batch_size, sequence_length, hidden_size)
        output = torch.gather(hidden, 1, bop_index_list) # expect batch_size * max_length * hidden_size

        emission_scores = self.theta_emit(output) # expect batchsize * max_length * num_labels
        max_length = emission_scores.shape[1]
        if fix_crf:
            transition_scores = self.transitions.unsqueeze(0).expand(max_length, self.num_labels, self.num_labels)
            transition_scores = transition_scores.unsqueeze(0).expand(batchsize, -1, self.num_labels, self.num_labels)
        else:
            transition_scores = self.theta_trans(output) # expect batchsize * max_length * num_labels**2

        psi0 = emission_scores[:, 0]
        psis = transition_scores[:, 1:].view(batchsize, -1, self.num_labels, self.num_labels) + emission_scores[:, 1:].unsqueeze(2)
        # expect batchsize * (max_length - 1) * num_labels**2

        return psi0, psis  

# no "labels_in_{context_length}" and labels_mask
def collate_fn(data):
    context_length = global_context_length
    data_dict = {}
    data_dict["article_id"], data_dict["window_start_index"], data_dict[f"{context_length}_chunk_tokens"], data_dict["attention_mask"] = [], [], [], []
    data_dict["context_len_for_prediction"], data_dict["start_in_context"], data_dict["end_in_context"], data_dict["min_two_sides"] = [], [], [], []
    data_dict["bop_index"], data_dict["labels"] = [], []
    # data_dict["bop_index"], data_dict["labels"], 

    max_length = 0
    for element in data:
        max_length = max(max_length, len(element["labels"]))
        assert len(data_dict["bop_index"]) == len(data_dict["labels"])
        for key, value in data_dict.items():
            value.append(element[key])
    new_labels_list_CE = []
    new_labels_list = []
    new_bop_index_list = []
    new_label_mask = []
    for idx in range(len(data)):
        curr_label_list = data[idx]["labels"]
        curr_length = len(curr_label_list)
        curr_bop_index = data[idx]["bop_index"]
        curr_label_mask = curr_length * [1]

        curr_label_list_CE = data[idx]["labels_CE"]
        curr_label_list_CE.extend(0 for k in range(max_length - curr_length))
        new_labels_list_CE.append(curr_label_list_CE)

        curr_label_list.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for k in range(max_length - curr_length))
        curr_bop_index.extend(0 for k in range(max_length - curr_length))
        curr_label_mask.extend(0 for k in range(max_length - curr_length))
        new_labels_list.append(curr_label_list)
        new_bop_index_list.append(curr_bop_index)
        new_label_mask.append(curr_label_mask)

    new_data_dict = {}
    for key in ["article_id", "window_start_index", "attention_mask",
    "context_len_for_prediction", "start_in_context", "end_in_context", "min_two_sides"]:
        new_data_dict[key] = torch.tensor(data_dict[key])
    new_data_dict["input_ids"] = torch.tensor(data_dict[f"{context_length}_chunk_tokens"])

    new_data_dict["labels_new_CE"] = torch.tensor(new_labels_list_CE) # expect batch_size * max_length
    new_data_dict["labels_new"] = torch.tensor(new_labels_list) # expect batch_size * max_length * 14
    new_data_dict["bop_index"] = torch.tensor(new_bop_index_list) # expect batch_size * max_length
    new_data_dict["new_label_mask"] = torch.tensor(new_label_mask) # expect batch_size * max_length
    return new_data_dict
def forward_alg(psi0, psis, new_label_mask):
    """
    Computes the forward algorithm.

    args
      - psi0: a tensor of real scores (which can be < 0), of dimension batchsize x K
      - psis: a tensor of real scores (which can be < 0), of dimension batchsize x M-1 x K x K
      - new_label_mask: batch_size * max_length
    returns
      - log_Zs: a batchsize tensor of the log of the sum of the possible scores for each x sequence
    """
    bsz, K = psi0.size() # batchsize, nlabels
    M = psis.size(1) + 1 # sequence length
    alpha = psi0 # bsz x K stores sum of all possible scores at CURRENT time step

    new_label_mask_length = torch.sum(new_label_mask, 1, keepdim=True) # expect batch_size * 1
    for m in range(1, M):
        alpha_temp = alpha.unsqueeze(2) #expect batchsize * K * 1
        alpha_temp = alpha_temp.expand(-1, -1, K) # expect batchsize * K * K
        currTable = psis[:, m - 1, :, :] # expect batchsize x K x K
        currTable = currTable + alpha_temp # expect batchsize x K x K
        alpha_temp = torch.logsumexp(currTable, 1) # expect batchsize * K

        continue_mask = new_label_mask_length > m # expect batch_size * 1
        alpha_temp = continue_mask * alpha_temp

        stop_mask = continue_mask == False
        alpha = alpha_temp + alpha * stop_mask # expect batchsize * K

    log_Zs = torch.logsumexp(alpha, dim=1) # bsz
    return log_Zs

def gold_sequence_scores(Y, psi0, psis, new_label_mask):
    """
    Calculate total scores of a batch of sequences:
        total_score(x, y) = sum_{m=1}^M psi(x, y_{m}, y_{m-1}, m)
    args
        - Y: batchsize x M matrix of label type indices
        - psi0: a tensor of real scores (which can be < 0), of dimension batchsize x K
        - psis: a tensor of real scores (which can be < 0), of dimension batchsize x M-1 x K x K
        - new_label_mask: batch_size * max_length
    N.B.: we don't need X as an argument, because psi0 and psis already
    condition on (i.e., know about) X
    """
    bsz, M = Y.size()
    # create indices [0...batchsize-1], which will be handy for indexing along 0th dimension
    all_example_idxs = torch.arange(bsz).to(Y.device)

    # calculate the true first label's score for each example
    scores = psi0[all_example_idxs, Y[:, 0]] # bsz
    # the above is equivalent to psi0.gather(1, Y[:, 0].unsqueeze(1)).view(-1)

    # add the scores for the remaining time-steps
    for m in range(1, M):
        psis_m = psis[:, m-1] # bsz x K x K
        # for each example get the label at the previous time step and at the current one
        transitions_m = Y[:, m-1:m+1] # bsz x 2
        # for each example add score corresponding to transitioning from label
        # y_{m-1} to label y_m
        scores = scores + psis_m[all_example_idxs, transitions_m[:, 0], transitions_m[:, 1]] * new_label_mask[:, m]

    return scores
def crf_nll(psi0, psis, Y, new_label_mask):
    """ 
    Calculate CRF negative log likelihood.

    args
        - Y: batchsize x M matrix of label type indices 
    returns
        - nlls - a batchsize length tensor of negative log likelihoods
    """
    # # get psi values for all time steps and all sequences in the batch
    # psi0, psis = self.get_sequence_psis(X)

    # score each sequence paired with its true label sequences
    gold_scores = gold_sequence_scores(Y, psi0, psis, new_label_mask) # batchsize

    # calculate log partition function for each example
    log_Zs = forward_alg(psi0, psis, new_label_mask) # batchsize

    ### Your CODE HERE!!!
    nlls = - gold_scores.sum() + log_Zs.sum()

    ### END YOUR CODE HERE
    return nlls
def train(model_name, context_length, batch_size, num_epochs, data_set, output_dir, model_num, gold_file, dropout=0, num_labels=14, learning_rate=5e-5, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=1e-5, random_seed=10, eval_frequency=25):
    torch.manual_seed(random_seed)
    global global_context_length
    global_context_length = context_length
    train_loader = DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(data_set['valid'], batch_size=batch_size, collate_fn=collate_fn)

    model = BERTClass(model_name, num_labels, dropout)
    print("Using learning rate", learning_rate)
    if fix_crf:
        optimizer = torch.optim.AdamW([{"params":model.roberta.parameters()},{"params":model.transitions}], lr=learning_rate, weight_decay=weight_decay) 
    else:
        optimizer = torch.optim.AdamW([{"params":model.roberta.parameters()},{"params":model.theta_emit.parameters()}], lr=learning_rate, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    model.to(device)
    model.train()

    class_frequency = [2123, 1058, 621, 466, 493, 294, 229, 209, 129, 144, 107, 76, 107, 72]
    max_freq = max(class_frequency)
    pos_weight = torch.tensor([max_freq/i for i in class_frequency]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    if turn_on_wandb:
        wandb.init(config={"init_epochs": num_epochs, "batch_size": batch_size}) # inside config.yaml
        wandb.run.name = output_dir
        wandb.run.save()
        wandb.watch(model, log_freq=10)
    step = 0
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # crf
            bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 1024).to(device) # expect batch_size * max_length * hidden_size
            psi0, psis = model(input_ids, attention_mask, bop_index_list)
            Y = batch['labels_new_CE'].to(device) # expect batch_size * max_length
            new_label_mask = batch['new_label_mask'].to(device)
            loss = crf_nll(psi0, psis, Y, new_label_mask)

            # # new
            # bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14).to(device) # expect batch_size * max_length * 14
            # output = torch.gather(output, 1, bop_index_list) # expect batch_size * max_length * 14
            # loss_in_full_dim = criterion(output, batch['labels_new'].float().to(device))
            # new_label_mask = batch['new_label_mask'].unsqueeze(-1).expand(-1, -1, num_labels).to(device) # expect batch_size * max_length * 14 
            # loss_in_full_dim = loss_in_full_dim * new_label_mask
            # loss = torch.sum(loss_in_full_dim) / torch.sum(new_label_mask)       

            total_loss += loss
            loss.backward()
            optimizer.step()

            if num_batches % eval_frequency == 0:
                print("loss = ", loss.item())
                model.eval()  
                model_eval.predict_and_write_to_file_for_chunk_crf(context_length, model, valid_loader, f"outputs/baseline-output-TC-{model_num}-{epoch}.txt")
                print(f"outputs/baseline-output-TC-{model_num}-{epoch}.txt")
                f1, precision, recall, f1_per_class, _, _ = model_eval.score(f"outputs/baseline-output-TC-{model_num}-{epoch}.txt", gold_file)
                log = {"loss": loss.item(), "step":step, "epoch":epoch, "F1": f1, "precision": precision, "recall": recall, "learning_rate": scheduler.get_last_lr()[0]}
                for i in range(14):
                    log[f"label{i}"] = f1_per_class[i]
                if turn_on_wandb: wandb.log(log)
                model.train()

            # early stopping 
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f'outputs/best_model_chunk-{random_seed}-{context_length}.pt') 

                with open(f"outputs/baseline-output-TC-{model_num}-{epoch}.txt",'r') as firstfile, open(f'outputs/pred_span-{random_seed}-{context_length}.txt','w') as secondfile:
                    for line in firstfile: secondfile.write(line)
            num_batches += 1
            step += 1
        scheduler.step()
    model.eval()
    if fix_crf:
        model_eval.write_best_f1_to_file(model.transitions, f'outputs/transitions-{random_seed}-{model_num}-{context_length}.txt')
    model_eval.write_best_f1_to_file(best_f1, f'outputs/best_f1_chunk-{random_seed}-{model_num}-{context_length}.txt')

from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel, LongformerConfig, LongformerModel
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch
import model_eval
# import wandb
turn_on_wandb = False
global_context_length = 0
# gold_file = "evaluation/gold/dev-task-flc-tc.labels.txt"
# gold_file = "split_data/datasets/3/dev-task-flc-tc.labels.txt"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BERTClass(torch.nn.Module):
    def __init__(self, model_name, num_labels, dropout):
        super(BERTClass, self).__init__()
        self.num_labels = num_labels
        if model_name == "longformer":
            self.project_down = torch.nn.Linear(768, num_labels)
            configuration = LongformerConfig.from_pretrained('allenai/longformer-base-4096', num_labels=num_labels, hidden_dropout_prob=dropout)
            self.roberta = LongformerModel.from_pretrained("allenai/longformer-base-4096", config=configuration)
        else: 
            self.project_down = torch.nn.Linear(1024, num_labels)
            configuration = RobertaConfig.from_pretrained('roberta-large', num_labels=num_labels, hidden_dropout_prob=dropout)
            self.roberta = RobertaModel.from_pretrained("roberta-large", config=configuration)
        self.roberta.resize_token_embeddings(50265 + 2 * 4)

    def forward(self, ids, mask):
        outputs = self.roberta(ids, mask) 
        hidden = outputs.last_hidden_state # (batch_size, sequence_length, hidden_size)

        x = self.project_down(hidden) # (batch_size, max_length, num_labels)
        return x 
# no "labels_in_{context_length}" and labels_mask
def collate_fn(data):
    context_length = global_context_length
    data_dict = {}
    data_dict["article_id"], data_dict["window_start_index"], data_dict[f"{context_length}_chunk_tokens"], data_dict["attention_mask"] = [], [], [], []
    data_dict["context_len_for_prediction"], data_dict["start_in_context"], data_dict["end_in_context"], data_dict["min_two_sides"] = [], [], [], []
    data_dict["bop_index"], data_dict["labels"] = [], []

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

    # for CE
    new_data_dict["labels_new_CE"] = torch.tensor(new_labels_list_CE) # expect batch_size * max_length

    new_data_dict["labels_new"] = torch.tensor(new_labels_list) # expect batch_size * max_length * 14
    new_data_dict["bop_index"] = torch.tensor(new_bop_index_list) # expect batch_size * max_length
    new_data_dict["new_label_mask"] = torch.tensor(new_label_mask) # expect batch_size * max_length
    return new_data_dict

def train(model_name, context_length, batch_size, num_epochs, data_set, output_dir, model_num, gold_file, dropout=0, num_labels=14, learning_rate=5e-5, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=1e-5, random_seed=10, eval_frequency=25):
    torch.manual_seed(random_seed)
    global global_context_length
    global_context_length = context_length
    # if model_name == "longformer":
    #     train_loader = DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_long)
    #     valid_loader = DataLoader(data_set['valid'], batch_size=batch_size, collate_fn=collate_fn_for_long)
    # else: 
    train_loader = DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(data_set['valid'], batch_size=batch_size, collate_fn=collate_fn)

    model = BERTClass(model_name, num_labels, dropout)
    print("Using learning rate", learning_rate)

    optimizer = torch.optim.AdamW(model.roberta.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    model.to(device)
    model.train()
    class_frequency = [2448, 1241, 766, 534, 559, 338, 316, 227, 169, 158, 129, 93, 136, 77]
    # class_frequency = [2123, 1058, 621, 466, 493, 294, 229, 209, 129, 144, 107, 76, 107, 72]
    max_freq = max(class_frequency)
    pos_weight = torch.tensor([max_freq/i for i in class_frequency])
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    criterion = torch.nn.CrossEntropyLoss(weight=pos_weight, reduction='none')
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
        with torch.autograd.set_detect_anomaly(True):
            for batch_index, batch in enumerate(train_loader):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                print(input_ids.shape)
                print(attention_mask.shape)
                assert False
                # labels_mask = batch['labels_mask'].to(device) # (batch_size, sequence_length)
                
                output = model(input_ids, attention_mask)
                bop_index_list = batch['bop_index'].unsqueeze(2).repeat(1, 1, 14) # expect batch_size * max_length * 14
                output = torch.gather(output.cpu(), 1, bop_index_list) # expect batch_size * max_length * 14


                # CE
                loss_in_full_dim = criterion(torch.transpose(output, 1, 2), batch["labels_new_CE"]) # expect batch_size * max_length
                new_label_mask = batch['new_label_mask']

                loss_in_full_dim = loss_in_full_dim * new_label_mask
                loss = torch.sum(loss_in_full_dim) / torch.sum(new_label_mask)    

                total_loss += loss
                loss.backward()
                optimizer.step()

                if num_batches % eval_frequency == 0:
                    print("loss = ", loss.item())
                    model.eval()
    
                    model_eval.predict_and_write_to_file_for_chunk_CUDA(context_length, model, valid_loader, f"outputs/baseline-output-TC-{model_num}-{epoch}.txt")
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

                    with open(f"outputs/baseline-output-TC-{model_num}-{epoch}.txt",'r') as firstfile, open(f'outputs/pred_span-{random_seed}-{model_num}-{context_length}.txt','w') as secondfile:
                        for line in firstfile: secondfile.write(line)
                num_batches += 1
                step += 1
                break

        break
        scheduler.step()
    model.eval()
    model_eval.write_best_f1_to_file(best_f1, f'outputs/best_f1_chunk-{random_seed}-{model_num}-{context_length}.txt')
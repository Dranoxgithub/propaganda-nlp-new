from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel, LongformerConfig, LongformerModel
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch
import model_eval
import wandb
turn_on_wandb = True
# gold_file = "evaluation/gold/dev-task-flc-tc.labels.txt"
# gold_file = "split_data/datasets/4/dev-task-flc-tc.labels.txt"
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
        self.roberta.resize_token_embeddings(50267)


        ### classification head
        self.dense = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.1)
        self.out_proj = torch.nn.Linear(1024, 14)
    
    def forward(self, ids, mask, bop_index):
        outputs = self.roberta(ids, mask) 
        hidden = outputs.last_hidden_state # expect batchsize * seq_length * 1024/ 768(longformer)
        x = torch.cat([ torch.index_select(a, 0, i) for a, i in zip(hidden, bop_index) ], dim=0)

        # x = self.project_down(x) # expect batchsize * num_labels

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# mainly for modification on input_ids and attention mask to pad to the longest of that batch
def collate_fn(data):
    data_dict = {}
    data_dict["article_id"], data_dict["start"], data_dict["end"], data_dict["label"], data_dict["bop_index"] = [], [], [], [], []
    
    max_length = 0
    for element in data:
        max_length = max(max_length, len(element["input_ids"]))
        for key, value in data_dict.items():
            value.append(element[key])
    
    new_input_ids_list = []
    new_attention_mask = []
    for idx in range(len(data)):
        curr_input_id = data[idx]["input_ids"]
        curr_length = len(curr_input_id)
        curr_attention_mask = curr_length * [1]

        curr_input_id.extend(0 for k in range(max_length - curr_length))
        curr_attention_mask.extend(0 for k in range(max_length - curr_length))
        new_input_ids_list.append(curr_input_id)
        new_attention_mask.append(curr_attention_mask)
        
    new_data_dict = {}
    for key in ['article_id', 'start', 'end', 'label', 'bop_index']:
        new_data_dict[key] = torch.tensor(data_dict[key])
    new_data_dict["input_ids"] = torch.tensor(new_input_ids_list)
    new_data_dict["attention_mask"] = torch.tensor(new_attention_mask)
    return new_data_dict

def train(model_name, context_length, batch_size, num_epochs, data_set, output_dir, model_num, gold_file, dropout=0, num_labels=14, learning_rate=5e-5, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=1e-5, random_seed=10, eval_frequency=25):
    torch.manual_seed(random_seed)
    train_loader = DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(data_set['valid'], batch_size=batch_size, collate_fn=collate_fn)
    # subtrain_loader = DataLoader(data_set['subtrain'], batch_size=batch_size)
    # for batch_index, batch in enumerate(train_loader):
    #     for key, value in batch.items():
    #         print(key)
    #         # print(value)
    #         print(value.shape)
    #     break
    model = BERTClass(model_name, num_labels, dropout)
    print("Using learning rate", learning_rate)
    optimizer = torch.optim.AdamW(model.roberta.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.2)
    model.to(device)
    model.train()

    accum_iter = 2 

    class_frequency = [2448, 1241, 766, 534, 559, 338, 316, 227, 169, 158, 129, 93, 136, 77]
    # [2123, 1058, 621, 466, 493, 294, 229, 209, 129, 144, 107, 76, 107, 72]
    max_freq = max(class_frequency)
    # pos_weight = torch.tensor([max_freq/i for i in class_frequency]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    if turn_on_wandb:
        wandb.init(config={"init_epochs": num_epochs, "batch_size": batch_size, "dropout_rate": dropout, 
            "weight_decay": weight_decay, "random_seed": random_seed, "context_length": context_length}) # inside config.yaml
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bop_index = batch['bop_index'].to(device)

            output = model(input_ids, attention_mask, bop_index)
            loss = criterion(output, batch['label'].float().to(device))

            # accumulated gradient
            loss = loss / accum_iter

            total_loss += loss 
            loss.backward()


            if ((batch_index + 1) % accum_iter == 0) or (batch_index + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            if num_batches % eval_frequency == 0:
                print("loss = ", loss.item())
                model.eval()
                dev_loss = model_eval.predict_and_write_to_file_for_span(model, valid_loader, f"outputs/baseline-output-TC-{model_num}-{epoch}.txt")
                print(f"outputs/baseline-output-TC-{model_num}-{epoch}.txt")
                f1, precision, recall, f1_per_class, _, _ = model_eval.score(f"outputs/baseline-output-TC-{model_num}-{epoch}.txt", gold_file)
                log = {"dev_loss": dev_loss, "loss": loss.item(), "step":step, "epoch":epoch, "F1": f1, "precision": precision, "recall": recall, "learning_rate": scheduler.get_last_lr()[0]}
                for i in range(14):
                    log[f"label{i}"] = f1_per_class[i]
                
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), f'outputs/best_model_span-{random_seed}-{model_num}-{context_length}.pt') 

                    with open(f"outputs/baseline-output-TC-{model_num}-{epoch}.txt",'r') as firstfile, open(f'outputs/pred_span-{random_seed}-{model_num}-{context_length}.txt','w') as secondfile:
                        for line in firstfile: secondfile.write(line)


                if turn_on_wandb: wandb.log(log)
                model.train()

            num_batches += 1
            step += 1
        scheduler.step()
    model.eval()

    print("-----------")
    best_state_dict = torch.load(f'outputs/best_model_span-{random_seed}-{model_num}-{context_length}.pt', map_location=device)
    model.load_state_dict(best_state_dict)
    model.eval()
    model_eval.predict_and_write_to_file_for_span(model, valid_loader, f'outputs/best_model_span-{random_seed}-{context_length}-test.txt')
    f1, precision, recall, f1_per_class, _, _ = model_eval.score(f'outputs/best_model_span-{random_seed}-{context_length}-test.txt', gold_file)
    print(f"best-f1: {best_f1}")
    model_eval.write_best_f1_to_file(best_f1, f'outputs/best_f1_span-{random_seed}-{model_num}-{context_length}.txt')
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel, LongformerConfig, LongformerModel, RobertaForSequenceClassification
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

        self.project_down = torch.nn.Linear(1024, num_labels)
        configuration = RobertaConfig.from_pretrained('roberta-large', num_labels=num_labels, hidden_dropout_prob=dropout, output_hidden_states=True)
        self.roberta = RobertaForSequenceClassification.from_pretrained("roberta-large", config=configuration)
        self.roberta.resize_token_embeddings(50267)
# output_hidden_states=True

        ### classification head, included in the RobertaForSequenceClassification
        self.dense = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.1)
        self.out_proj = torch.nn.Linear(1024, 14)

        self.classifier0 = torch.nn.Linear(1024, 2)
        self.classifier1 = torch.nn.Linear(1024, 2)
        self.classifier2 = torch.nn.Linear(1024, 4)
        self.classifier3 = torch.nn.Linear(1024, 3)
        self.classifier4 = torch.nn.Linear(1024, 2)
        self.classifier5 = torch.nn.Linear(1024, 4)
        self.classifier6 = torch.nn.Linear(1024, 3)
    
    def forward(self, ids, mask, bop_index):
        outputs = self.roberta(ids, mask)
        x = outputs.logits # torch.Size([8, 14]) batch_size * num_labels
        hidden = outputs.hidden_states[-1] # last_hidden_state, batch_size * seq_length * hidden_size
        
        return x, [self.classifier0(hidden), self.classifier1(hidden), self.classifier2(hidden), self.classifier3(hidden), self.classifier4(hidden), self.classifier5(hidden), self.classifier6(hidden)]
label_to_mask = {0: [1, 0, 0, 1, 1, 0, 1], # 0, 3, 4, 6
                3: [1, 0, 0, 1, 1, 0, 1], # 0, 3, 4, 6
                6: [1, 0, 0, 1, 1, 0, 1], # 0, 3, 4, 6
                1: [1, 0, 0, 1, 0, 0, 0], # 0, 3
                2: [1, 0, 0, 1, 1, 1, 0], # 0, 3, 4, 5
                4: [1, 0, 0, 1, 1, 1, 0], # 0, 3, 4, 5
                5: [1, 0, 0, 1, 1, 1, 0], # 0, 3, 4, 5
                8: [1, 0, 0, 1, 1, 1, 0], # 0, 3, 4, 5
                13: [1, 0, 0, 1, 0, 0, 0], # 0, 3
                12: [1, 1, 0, 0, 0, 0, 0], # 0, 1
                7: [1, 1, 1, 0, 0, 0, 0], # 0, 1, 2
                9: [1, 1, 1, 0, 0, 0, 0], # 0, 1, 2
                10: [1, 1, 1, 0, 0, 0, 0], # 0, 1, 2
                11: [1, 1, 1, 0, 0, 0, 0]} # 0, 1, 2

def classfier_encoding(classifier_num, label):
    def helper(label, candidate_list):
        ret = [0] * len(candidate_list)
        ret[candidate_list.index(label)] = 1
        return ret
    if classifier_num == 0:
        if label in {7, 9, 10, 11, 12}:
            return [1, 0]
        else: return [0, 1]

    if classifier_num == 1:
        if label == 12: return [1, 0]
        elif label in {7, 9, 10, 11}: return [0, 1]
        else: return [0, 0]
    
    if classifier_num == 2:
        if label in {7, 9, 10, 11}: return helper(label, [7, 9, 10, 11])
        else: return [0, 0, 0, 0]
    
    if classifier_num == 3:
        if label == 1: return [1, 0, 0]
        elif label in {2, 4, 5, 8, 0, 3, 6}: return [0, 1, 0]
        elif label == 13: return [0, 0, 1]
        else: return [0, 0, 0]
    
    if classifier_num == 4:
        if label in {2, 4, 5, 8}: return [1, 0]
        elif label in {0, 3, 6}: return [0, 1]
        else: return [0, 0]
    
    if classifier_num == 5:
        if label in {2, 4, 5, 8}: return helper(label, [2, 4, 5, 8])
        else: return [0, 0, 0, 0]
    
    if classifier_num == 6:
        if label in {0, 3, 6}: return helper(label, [0, 3, 6])
        else: return [0, 0, 0]
classifier_num_label = [2, 2, 4, 3, 2, 4, 3]
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
    new_labels_list_CE = []
    mask_map = {}
    classifier_encoding_map = {}

    # innitialization
    for classifier_num in range(7):
        mask_map[f'classifier_mask_{classifier_num}'] = []
        classifier_encoding_map[f'classifier_{classifier_num}'] = []

    for idx in range(len(data)):
        curr_input_id = data[idx]["input_ids"]
        curr_length = len(curr_input_id)
        curr_attention_mask = curr_length * [1]

        curr_input_id.extend(0 for k in range(max_length - curr_length))
        curr_attention_mask.extend(0 for k in range(max_length - curr_length))
        new_input_ids_list.append(curr_input_id)
        new_attention_mask.append(curr_attention_mask)

        # get the rarer label
        label_num = max(index for index, item in enumerate(data[idx]["label"]) if item == 1)
        new_labels_list_CE.append(label_num)
        for classifier_num in range(7):
            curr_classfier_encoding = []
            # change encoding
            # shuffle
            # label_num = shuffle(label_num)

            curr_classifier_mask = label_to_mask[label_num][classifier_num]
            one_hot_encoding = classfier_encoding(classifier_num, label_num)
                
            # CE
            encoding = one_hot_encoding.index(1) if 1 in one_hot_encoding else 0
            
            mask_map[f'classifier_mask_{classifier_num}'].append(curr_classifier_mask)
            classifier_encoding_map[f'classifier_{classifier_num}'].append(encoding)

        
    new_data_dict = {}
    for key in ['article_id', 'start', 'end', 'label', 'bop_index']:
        new_data_dict[key] = torch.tensor(data_dict[key])
    new_data_dict["input_ids"] = torch.tensor(new_input_ids_list)
    new_data_dict["attention_mask"] = torch.tensor(new_attention_mask)

    # for CE
    new_data_dict["labels_new_CE"] = torch.tensor(new_labels_list_CE) # expect batch_size * max_length
    for classifier_num in range(7):
        new_data_dict[f'classifier_mask_{classifier_num}'] = torch.tensor(mask_map[f'classifier_mask_{classifier_num}'])
        new_data_dict[f'classifier_{classifier_num}'] = torch.tensor(classifier_encoding_map[f'classifier_{classifier_num}'])

    return new_data_dict

def train(model_name, context_length, batch_size, num_epochs, data_set, output_dir, model_num, aux_weight_train, aux_weight_eval, gold_file, dropout=0, num_labels=14, learning_rate=5e-5, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=1e-5, random_seed=10, eval_frequency=25):
    torch.manual_seed(random_seed)
    train_loader = DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(data_set['valid'], batch_size=batch_size, collate_fn=collate_fn)
    # subtrain_loader = DataLoader(data_set['subtrain'], batch_size=batch_size)
    # for batch_index, batch in enumerate(train_loader):
    #     for key, value in batch.items():
    #         print(key)
    #         # print(value)
    #         print(value)

    model = BERTClass(model_name, num_labels, dropout)
    print("Using learning rate", learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 
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

            # output: batch_size * seq_length(512) * num_labels_for_classifier
            # bop_index: not used, just take the first one since we are using the head method
            # expected: batch_size
            # mask: batch_size
            def compute_auxiliary_loss(output, bop_index, expected, mask):
                criterion = torch.nn.CrossEntropyLoss(reduction='none')
                num_labels = output.shape[2]
             
               
                output = output[:, 0, :] # expect batch_size  * num_labels_for_classifier

                loss_in_full_dim = criterion(output, expected.to(device)) # expect batch_size
                new_label_mask = mask.to(device) # expect batch_size
                    
                loss_in_full_dim = loss_in_full_dim * new_label_mask
                return loss_in_full_dim 
            output, test = model(input_ids, attention_mask, bop_index)
            ori_loss = criterion(output, batch['label'].float().to(device))

            # accumulated gradient
            ori_loss = ori_loss / accum_iter

            num_active_classifier_mask = batch['classifier_mask_0']
            aux_loss_full_dim = compute_auxiliary_loss(test[0], bop_index, batch['classifier_0'], batch['classifier_mask_0'])
            for classifier_num in range(1, 7):
                num_active_classifier_mask += batch[f'classifier_mask_{classifier_num}']
                aux_loss_full_dim += compute_auxiliary_loss(test[classifier_num], bop_index, batch[f'classifier_{classifier_num}'], batch[f'classifier_mask_{classifier_num}'])
            classifier_mask = (num_active_classifier_mask > 0) * 1
            # change 0 to 1 in num_active_classifier_mask so as not to create nan when dividing
            num_active_classifier_mask[num_active_classifier_mask == 0] = 1
            avg_aux_loss_full_dim = aux_loss_full_dim / num_active_classifier_mask.to(device)
            auxiliary_loss = torch.sum(avg_aux_loss_full_dim) / torch.sum(classifier_mask)
            auxiliary_loss = auxiliary_loss / accum_iter

            loss = ori_loss * (1 - aux_weight_train) + aux_weight_train * auxiliary_loss

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
                dev_loss = model_eval.predict_and_write_to_file_for_span_hier(model, valid_loader, f"outputs/baseline-output-TC-{model_num}-{epoch}.txt", aux_weight_eval)
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
    model_eval.predict_and_write_to_file_for_span_hier(model, valid_loader, f'outputs/best_model_span-{random_seed}-{model_num}-{context_length}-test.txt', aux_weight_eval)
    f1, precision, recall, f1_per_class, _, _ = model_eval.score(f'outputs/best_model_span-{random_seed}-{model_num}-{context_length}-test.txt', gold_file)
    print(f"best-f1: {best_f1}")
    model_eval.write_best_f1_to_file(f1, f'outputs/best_f1_span-{random_seed}-{model_num}-{context_length}.txt')
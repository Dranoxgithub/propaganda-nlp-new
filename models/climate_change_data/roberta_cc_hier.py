from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel, AdamW
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch
import wandb
import numpy as np
turn_on_wandb = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def label_to_mask(label, sublabel):
#     mask = [0] * 6
#     mask[0] = 1
#     mask[label + 1] = 1
#     return mask

def write_best_f1_to_file(best_f1, file_name):
    with open(file_name, "w") as fout:
        fout.write(f"best_f1: {best_f1}")
def predict_and_write_to_file_for_cc_hier(model, loader, output_file_name, aux_weight):
    model.eval()
    total_loss = 0

    # class_frequency = [2448, 1241, 766, 534, 559, 338, 316, 227, 169, 158, 129, 93, 136, 77]
    # max_freq = max(class_frequency)
    # pos_weight = torch.tensor([max_freq/i for i in class_frequency]).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    num_batches = 0
    with torch.no_grad() and open(output_file_name, "w") as fout:
        prediction = []
        label = []
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            output, test = model(input_ids, mask)
            aux_prob = compute_auxiliary_prob(test)
            combi_prob = aux_prob * aux_weight + output * (1 - aux_weight)

            total_loss += criterion(output, batch["label"].to(device)).item()
            num_batches += 1 

            max_prob, max_index = torch.topk(combi_prob, 1, dim = 1) # expect batchsize * 1
            for i in range(batch['input_ids'].shape[0]):
                prediction.append(max_index[i][0].item())
                label.append(batch["label"][i].item())
                fout.write(f"{batch['paragraph_index'][i]}\t{max_index[i][0]}\n")
    return total_loss / num_batches, f1_score(label, prediction, average='macro'), precision_score(label, prediction, average='macro'), recall_score(label, prediction, average='macro')

class BERTClass(torch.nn.Module):
    def __init__(self, num_labels):
        super(BERTClass, self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained('roberta-large', num_labels=num_labels)
        # self.roberta.resize_token_embeddings(50267)
        self.project_down = torch.nn.Linear(1024, num_labels)

        self.classifier0 = torch.nn.Linear(1024, 5)
        self.classifier1 = torch.nn.Linear(1024, 8)
        self.classifier2 = torch.nn.Linear(1024, 5)
        self.classifier3 = torch.nn.Linear(1024, 6)
        self.classifier4 = torch.nn.Linear(1024, 5)
        self.classifier5 = torch.nn.Linear(1024, 3)
    
    def forward(self, ids, mask):
        outputs = self.roberta(ids, mask) 
        hidden = outputs.last_hidden_state # expect batchsize * seq_length * 1024
        x = hidden[:, 0, :] # expect batchsize * 1024
        return self.project_down(x), [self.classifier0(x), self.classifier1(x), self.classifier2(x), self.classifier3(x), self.classifier4(x), self.classifier5(x)]
def compute_auxiliary_prob(test):
    output0, output1, output2, output3, output4, output5 = test
    # output0[:, 0] expect batchsize
    test0 = output0[:, 0].unsqueeze(1).expand(-1, output1.shape[1]) * output1
    test1 = output0[:, 1].unsqueeze(1).expand(-1, output2.shape[1]) * output2
    test2 = output0[:, 2].unsqueeze(1).expand(-1, output3.shape[1]) * output3
    test3 = output0[:, 3].unsqueeze(1).expand(-1, output4.shape[1]) * output4
    test4 = output0[:, 4].unsqueeze(1).expand(-1, output5.shape[1]) * output5


    return torch.cat((test0, test1, test2, test3, test4), 1)
def collate_fn(data):
    data_dict = {}
    data_dict["paragraph_index"], data_dict["label"], data_dict["sublabel"] = [], [], []
    
    max_length = 0
    for element in data:
        max_length = max(max_length, len(element["input_ids"]))
        for key, value in data_dict.items():
            value.append(element[key])
    new_input_ids_list = []
    new_attention_mask = []

    mask_map = {}
    classifier_encoding_map = {}

    # innitialization
    for classifier_num in range(6):
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

        label = data[idx]["label"]
        sublabel = data[idx]["sublabel"]
        classifier_encoding_map[f'classifier_0'].append(label)
        mask_map[f'classifier_mask_0'].append(1)
        for classifier_num in range(1, 6):
            if label + 1 == classifier_num:
                mask_map[f'classifier_mask_{classifier_num}'].append(1)
                classifier_encoding_map[f'classifier_{classifier_num}'].append(sublabel)
            else:
                mask_map[f'classifier_mask_{classifier_num}'].append(0)
                classifier_encoding_map[f'classifier_{classifier_num}'].append(0)
    new_data_dict = {}

    for classifier_num in range(6):
        new_data_dict[f'classifier_mask_{classifier_num}'] = torch.tensor(mask_map[f'classifier_mask_{classifier_num}'])
        new_data_dict[f'classifier_{classifier_num}'] = torch.tensor(classifier_encoding_map[f'classifier_{classifier_num}'])

    for key in ['paragraph_index', 'label', 'sublabel']:
        new_data_dict[key] = torch.tensor(data_dict[key])
    new_data_dict["input_ids"] = torch.tensor(new_input_ids_list)
    new_data_dict["attention_mask"] = torch.tensor(new_attention_mask)
    return new_data_dict
def train(model_name, context_length, batch_size, num_epochs, data_set, output_dir, model_num, aux_weight_train, aux_weight_eval, dropout=0, num_labels=27, learning_rate=5e-5, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=1e-5, random_seed=10, eval_frequency=25):
    torch.manual_seed(random_seed)
    print("entered successfully")
    train_loader = DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(data_set['valid'], batch_size=batch_size, collate_fn=collate_fn)

    model = BERTClass(num_labels = num_labels)
    print("Using learning rate", learning_rate)
    optimizer = torch.optim.AdamW(model.roberta.parameters(), lr=learning_rate, weight_decay=weight_decay) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    model.to(device)
    model.train()

    # class_frequency = [2448, 1241, 766, 534, 559, 338, 316, 227, 169, 158, 129, 93, 136, 77]
    # max_freq = max(class_frequency)
    # pos_weight = torch.tensor([max_freq/i for i in class_frequency]).to(device)
    # normalized_pos_weight = torch.nn.functional.normalize(pos_weight, dim=0)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    if turn_on_wandb:
        wandb.init(config={"init_epochs": num_epochs, "batch_size": batch_size, "dropout_rate": dropout, 
            "weight_decay": weight_decay, "random_seed": random_seed, "context_length": context_length}) 
        wandb.run.name = output_dir
        wandb.run.save()
        wandb.watch(model, log_freq=10)
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output, test = model(input_ids, attention_mask)
            
            def compute_auxiliary_loss(output, expected, mask):
                # CE
                # weight_tensor = torch.tensor(weight).to(device).float()
                criterion = torch.nn.CrossEntropyLoss(reduction='none')

                loss_in_full_dim = criterion(output, expected.to(device)) # expect batch_size

                new_label_mask = mask.to(device) # expect batch_size
                    
                loss_in_full_dim = loss_in_full_dim * new_label_mask
                return loss_in_full_dim 


            num_active_classifier_mask = batch['classifier_mask_0']
            aux_loss_full_dim = compute_auxiliary_loss(test[0], batch['classifier_0'], batch['classifier_mask_0'])
            for classifier_num in range(1, 6):
                num_active_classifier_mask += batch[f'classifier_mask_{classifier_num}']
                aux_loss_full_dim += compute_auxiliary_loss(test[classifier_num], batch[f'classifier_{classifier_num}'], batch[f'classifier_mask_{classifier_num}']) # [1] * classifier_num_label[classifier_num]
            classifier_mask = (num_active_classifier_mask > 0) * 1
            # change 0 to 1 in num_active_classifier_mask so as not to create nan when dividing
            num_active_classifier_mask[num_active_classifier_mask == 0] = 1
            avg_aux_loss_full_dim = aux_loss_full_dim / num_active_classifier_mask.to(device)
            auxiliary_loss = torch.sum(avg_aux_loss_full_dim) / torch.sum(classifier_mask)

            ori_loss = criterion(output, batch["label"].to(device))
            loss = ori_loss * (1 - aux_weight_train) + aux_weight_train * auxiliary_loss
            total_loss += loss
            loss.backward()
            optimizer.step()
            if num_batches % eval_frequency == 0:
                print("loss = ", loss.item())
                model.eval()
                _, f1, precision, recall  = predict_and_write_to_file_for_cc_hier(model, valid_loader, f"outputs/roberta_cc_hier-{model_num}-{epoch}.txt", aux_weight_eval)


                log = {"loss": loss.item(), "epoch":epoch, "F1": f1, "precision": precision, "recall": recall, "learning_rate": scheduler.get_last_lr()}
                # for i in range(14):
                #     log[f"label{i}"] = f1_per_class[i]
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), f'outputs/roberta_cc_hier-{random_seed}-{model_num}-{context_length}.pt') 

                    with open(f"outputs/roberta_cc_hier-{model_num}-{epoch}.txt",'r') as firstfile, open(f'outputs/roberta_cc_hier-{random_seed}-{model_num}-{context_length}.txt','w') as secondfile:
                        for line in firstfile: secondfile.write(line)

                if turn_on_wandb: wandb.log(log)
                
                model.train()
            num_batches += 1
        scheduler.step()
    model.eval()
    # best_state_dict = torch.load(f'outputs/best_model_span-{random_seed}-{model_num}-{context_length}.pt', map_location=device)
    # model.load_state_dict(best_state_dict)
    # model.eval()
    # model_eval.predict_and_write_to_file_for_span(model, valid_loader, f'outputs/best_model_span-{random_seed}-{context_length}-test.txt')
    # f1, precision, recall, f1_per_class, _, _ = model_eval.score(f'outputs/best_model_span-{random_seed}-{context_length}-test.txt', gold_file)
    print(f"best-f1: {best_f1}")
    write_best_f1_to_file(best_f1, f'outputs/best_f1_cc_hier-{random_seed}-{model_num}-{context_length}.txt')
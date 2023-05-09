from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaModel, LongformerConfig, LongformerModel
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch
import model_eval
turn_on_wandb = False
global_context_length = 0
# gold_file = "evaluation/gold/dev-task-flc-tc.labels.txt"
# gold_file = "split_data/datasets/2/dev-task-flc-tc.labels.txt"
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
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier0 = torch.nn.Linear(1024, 2)
        self.classifier1 = torch.nn.Linear(1024, 2)
        self.classifier2 = torch.nn.Linear(1024, 4)
        self.classifier3 = torch.nn.Linear(1024, 3)
        self.classifier4 = torch.nn.Linear(1024, 2)
        self.classifier5 = torch.nn.Linear(1024, 4)
        self.classifier6 = torch.nn.Linear(1024, 3)

    def forward(self, ids, mask):
        outputs = self.roberta(ids, mask) 
        hidden = outputs.last_hidden_state # (batch_size, sequence_length, hidden_size)

        x = self.project_down(hidden) # (batch_size, sequence_length, num_labels)
        x = self.dropout(x)
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
def shuffle(label):
    shuffle_list = [11, 12 , 1 , 7, 5, 2, 8, 4, 13, 6, 10, 0, 3, 9]
    return shuffle_list[label]
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
    # CE
    new_labels_list_CE = []
    new_labels_list = []
    new_bop_index_list = []
    new_label_mask = []
    mask_map = {}
    classifier_encoding_map = {}

    # innitialization
    for classifier_num in range(7):
        mask_map[f'classifier_mask_{classifier_num}'] = []
        classifier_encoding_map[f'classifier_{classifier_num}'] = []

    # data passed in a batch, loop through every batch element
    for idx in range(len(data)):
        curr_label_list = data[idx]["labels"]
        curr_length = len(curr_label_list)
        curr_bop_index = data[idx]["bop_index"]
        curr_label_mask = curr_length * [1]
        # CE
        curr_label_list_CE = data[idx]["labels_CE"]

        for classifier_num in range(7):
            curr_classifier_mask = []
            curr_classfier_encoding = []
            for label_num in curr_label_list_CE: # change encoding
                # shuffle
                # label_num = shuffle(label_num)

                curr_classifier_mask.append(label_to_mask[label_num][classifier_num])
                one_hot_encoding = classfier_encoding(classifier_num, label_num)
                # CE
                encoding = one_hot_encoding.index(1) if 1 in one_hot_encoding else 0
                curr_classfier_encoding.append(encoding)
            curr_classifier_mask.extend(0 for k in range(max_length - curr_length))
            # CE
            curr_classfier_encoding.extend(0 for k in range(max_length - curr_length))
            # curr_classfier_encoding.extend(([0] * classifier_num_label[classifier_num]) for k in range(max_length - curr_length))
            mask_map[f'classifier_mask_{classifier_num}'].append(curr_classifier_mask)
            classifier_encoding_map[f'classifier_{classifier_num}'].append(curr_classfier_encoding)

        # CE
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
    for classifier_num in range(7):
        new_data_dict[f'classifier_mask_{classifier_num}'] = torch.tensor(mask_map[f'classifier_mask_{classifier_num}'])
        new_data_dict[f'classifier_{classifier_num}'] = torch.tensor(classifier_encoding_map[f'classifier_{classifier_num}'])

    new_data_dict["labels_new"] = torch.tensor(new_labels_list) # expect batch_size * max_length * 14
    new_data_dict["bop_index"] = torch.tensor(new_bop_index_list) # expect batch_size * max_length
    new_data_dict["new_label_mask"] = torch.tensor(new_label_mask) # expect batch_size * max_length
    return new_data_dict
freq = {0: [743, 6448], 1: [607, 136], 2: [227, 158, 129, 93], 3: [1241, 6448, 77], 4: [1832, 3296], 5: [766, 559, 338, 169], 6:[2448, 534, 316]}
def train(model_name, context_length, batch_size, num_epochs, data_set, output_dir, model_num, aux_weight_train, aux_weight_eval, gold_file, dropout=0, num_labels=14, learning_rate=5e-5, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=1e-5, random_seed=10, eval_frequency=25):
    torch.manual_seed(random_seed)
    global global_context_length
    global_context_length = context_length

    train_loader = DataLoader(data_set['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(data_set['valid'], batch_size=batch_size, collate_fn=collate_fn)

    model = BERTClass(model_name, num_labels, dropout)
    model.to(device)
    model.eval()
  
    # #
    model.load_state_dict(torch.load(f'outputs/best_model_chunk-{random_seed}-{model_num}-{context_length}-0.pt'))
   
    # threshold_list = [x * 0.1 for x in range(0, 11)]
    threshold_list = [0]
    print(threshold_list)
    for threshold in threshold_list:
        model_eval.predict_and_write_to_file_for_chunk_hier_early_stopping_1(context_length, model, valid_loader, f"outputs/hier-early-stopping-{model_num}-{threshold}-test1", 0, threshold)
        model_eval.sort_before_matching_1(f"outputs/hier-early-stopping-{model_num}-{threshold}-test1")


    # model_eval.predict_and_write_to_file_for_chunk_hier(context_length, model, valid_loader, f"outputs/hier-no-early-stopping-{model_num}", 0)
    # model_eval.sort_before_matching(f"outputs/hier-no-early-stopping-{model_num}")
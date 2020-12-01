import torch
import numpy as np, json
import matplotlib.pyplot as plt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
import pandas as pd
import importlib
from model.MedClinical import Biobert_fc 
from sklearn import metrics

from tqdm import tqdm_notebook, trange
import os
from transformers import BertTokenizer, BertModel
from transformers import  BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification
from transformers.optimization import AdamW
from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef, confusion_matrix


def eval_model(config, model, eval_dataloader, device,num_labels):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    labels = []
    inputs=[]
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        weights = torch.Tensor(config.hyperparams.LOSS_FN_CLASS_WEIGHTS)
        
        if ((device == 'cuda') or (device == 'cuda2')):
            class_weights = torch.FloatTensor(weights).cuda()
        else:
            class_weights = torch.FloatTensor(weights)
        loss_fct = CrossEntropyLoss(weight=class_weights, reduction='mean')
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().numpy())
            inputs.append(input_ids.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            labels[0] = np.append(
                labels[0], label_ids.detach().cpu().numpy(), axis=0)
            inputs[0] = np.append(
                inputs[0], input_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    
    preds = preds[0]
    labels = labels[0]
    inputs = inputs[0]
    
    preds = np.argmax(preds, axis=1)
    return(inputs, preds, labels, eval_loss)

def calculate_stats(labels, preds):

    
    CONFIG_FOLDER = 'config/'
    id_label_file = 'id_2_label.json'
    
    print('\n label:', labels)
    print('\n preds:', preds)
    
    with open(CONFIG_FOLDER + id_label_file) as infile:
        id2label = json.load(infile)    
        
    preds_labels = [id2label[str(p)] for p in preds]
    all_labels =  [id2label[str(l)] for l in labels]
    mcc = matthews_corrcoef(all_labels, preds_labels)
    
    mismatches = []
    all_rels = []
    for row in range(len(all_labels)):
        all_rels.append([all_labels[row], preds_labels[row]])
        if preds_labels[row] != all_labels[row]:
            mismatches.append([all_labels[row], preds_labels[row]])
    
    df_results = pd.DataFrame(all_rels, columns = ['labels', 'predicted'])
    
    
    f1_score = metrics.f1_score(df_results["labels"], df_results["predicted"], average='macro')
    
    df_results["matched"] = df_results["labels"] == df_results["predicted"]
    label_matches_df = df_results.groupby(["labels", "matched"]).count()
    
    return mcc, f1_score, df_results, label_matches_df


def get_BERT_Embedings(config, model, eval_dataloader, device,num_labels):
    bert_embedings = []
    labels = []
    model.eval()
    for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
   
#         print(type(logits))
#         print(logits)

        if(len(bert_embedings)  == 0):
                bert_embedings = logits
                labels.append(label_ids.detach().cpu().numpy())
        else:
                bert_embedings =torch.cat((torch.Tensor(bert_embedings), logits),0)
                labels[0] = np.append(
                    labels[0], label_ids.detach().cpu().numpy(), axis=0)        
        
    return(list(bert_embedings), labels)
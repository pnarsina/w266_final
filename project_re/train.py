import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
from torch import Tensor

from model.MedClinical import Biobert_fc 
from tqdm import tqdm_notebook, trange
import os
from transformers import BertTokenizer, BertModel
# from transformers import  BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification
from pytorch_pretrained_bert import  BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification
from transformers.optimization import AdamW
from torch.optim import lr_scheduler
import torch.optim as optim
from datetime import datetime
import pickle

def train_model(config, model,optimizer, scheduler, train_dataloader,  num_labels , data_len, device='cpu', model_save_path = "outputs", 
                model_name = 'BioBert_fc',      num_epochs=25, GRADIENT_ACCUMULATION_STEPS = 1 ):

    model = model.to(device)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_acc = 0
    model.train()
    metrics = []
    for epoch in trange(int(num_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        running_loss = 0.0
        running_corrects = 0
        epoch_acc = 0
        
        for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
            scheduler.step()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask)


            weights = torch.Tensor(config.hyperparams.LOSS_FN_CLASS_WEIGHTS)
            
            if(device =='cuda' or device =='cuda2'):
                class_weights = torch.FloatTensor(weights).cuda()
            else:
                class_weights = torch.FloatTensor(weights)
                
            loss_fct = CrossEntropyLoss(weight=class_weights, reduction='mean')

            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            preds = torch.argmax(logits, axis=1)
            if config.programsettings.DEBUG_PRINT == 1:
                print('\n predicted:', preds,  '\n true: ', label_ids.view(-1) )
            
            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            if config.programsettings.DEBUG_PRINT == 1:
                print("\r loss %f" %  loss, end='')

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            running_loss += loss.item() * nb_tr_examples
            running_corrects += torch.sum(preds == label_ids.view(-1))
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
            if config.programsettings.DEBUG_PRINT == 1:
                print('\n Accuumulated for ephoch, loss: ', running_loss, ' , corrects:', running_corrects, ' size: ', data_len)

        epoch_acc = np.double(running_corrects)/ (data_len)
        metrics.append([epoch, running_loss, epoch_acc ])
        
        if config.programsettings.DEBUG_PRINT == 1:
            print('epoch: {:d}  Acc: {:.4f}'.format(
                epoch,  epoch_acc))

        if  (epoch_acc > best_acc ):
            print("so far epoch accuracy: ", epoch_acc)
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
#     if config.programsettings.DEBUG_PRINT == 1:
    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    savemodel(model_name,model_save_path, model, metrics )
    
    return model

def savemodel(model_name, path, model, metrics):

    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + model_name)

    date_time_string = str(datetime.now()).replace(":", "_").replace(".", "_") 
    
    file_name = path + model_name + date_time_string + ".bin"
    torch.save(model, file_name)    
    
    metrics_file = path + model_name + date_time_string +  '_train_metrics_' + ".pkl"
    
    with open(metrics_file, "wb") as f:
        pickle.dump(metrics, f)      
import torch
import pickle, copy, json

from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss
import importlib
from model.MedClinical import Biobert_fc, Biobert_cnn_fc 
from train import *
from eval import eval_model, calculate_stats

from tqdm.notebook import tqdm, trange
import os
from transformers import BertTokenizer, BertModel
# from transformers import  BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification
from pytorch_pretrained_bert import  BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification
from transformers.optimization import AdamW
from torch.optim import lr_scheduler
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from util.tools  import *
from util.DataLoader import MultiClassificationProcessor




def run_model(config, device):
    model_config1 = model_config(config.modelconfig)
    
#     Load data and create features. If calculated features available in the cache, it will use it
    dataprocessor = MultiClassificationProcessor()
    train_dataloader, data_len, num_labels, num_train_optimization_steps, all_label_ids = dataprocessor.get_data_loader(config)

#   set to right Model Name
    if config.programsettings.MODEL_NAME == "BioBERT_fc":
        model = Biobert_fc(device, model_config1)
    if config.programsettings.MODEL_NAME == "BioBERT_CNN_fc":
        model = Biobert_cnn_fc(device, model_config1)
    elif config.programsettings.MODEL_NAME == "BERT_Sequence":
        model = BertForSequenceClassification.from_pretrained(config.programsettings.BERT_MODEL, cache_dir=config.programsettings.CACHE_DIR, num_labels=num_labels)

        
#   Freeze BERT layers if we don't want to tune based on configuraiton
    if config.hyperparams.NUM_BERT_LAYERS_FREEZE >= 0:
        count = 0 
        for child in model.children():
            if count > 0 and count < config.hyperparams.NUM_BERT_LAYERS_FREEZE:
                for param in child.parameters():
                    param.requires_grad = False
            count+=1
                    
# set the optimizer
    optimizer     = optim.AdamW(model.parameters(), lr=config.hyperparams.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.hyperparams.NUM_WARMP_STEPS, num_training_steps =num_train_optimization_steps)  # PyTorch scheduler

# Run the model
# from train import train_model
    model = train_model(config, model,  optimizer, scheduler, train_dataloader, num_labels, data_len,
                        device = device, model_save_path = config.programsettings.OUTPUT_DIR, 
                        model_name = config.programsettings.MODEL_NAME , num_epochs=config.hyperparams.NUM_TRAIN_EPOCHS)
    
# Evaluate training data
    train_inputs, train_preds, train_labels, train_loss = eval_model( config, model, train_dataloader, device, num_labels)

#   Prepare dev dataset
    dev_dataloader, dev_data_len, dev_num_labels, dev_num_train_optimization_steps, all_dev_label_ids = dataprocessor.get_data_loader(config,source='dev')    
#   Run the trained model on dev data    
    dev_inputs, dev_preds, dev_labels, dev_loss = eval_model( config, model, dev_dataloader, device, num_labels)    
    
    
# Evaluate Test data
    dataprocessor = MultiClassificationProcessor()
    test_dataloader, dev_data_len, dev_num_labels, dev_num_train_optimization_steps, all_dev_label_ids = dataprocessor.get_data_loader(config,source='test')    

    test_inputs, test_preds, test_labels, test_loss = eval_model( config, model, test_dataloader, device, num_labels)  
    
    return train_inputs, train_labels, train_preds, train_loss, dev_inputs, dev_labels, dev_loss, dev_preds,test_inputs, test_preds, test_labels, test_loss
import torch.nn as nn
import math
from transformers import BertModel
import torch
from .model_config import model_config 
import logging
import time
from platform import python_version
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable



class Biobert_fc(nn.Module):
    
    def __init__(self, device):
        super(Biobert_fc, self).__init__()

        self.model_conf = model_config()
      
        if device == 'cuda2':  
            self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))
            self.linear1 = nn.DataParallel(nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes))
        else:
            self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
            self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)
        
    def forward(self, ids, segment_ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               token_type_ids  = segment_ids,
               attention_mask=mask)
 
          linear1_output  = self.linear1(sequence_output[:,0,:].view(-1,self.model_conf.bert_features)) ## extract the 1st token's embeddings
 
#           linear2_output = self.linear2(linear1_output)
 
          return linear1_output

   
    
    
class Biobert_cnn_fc(nn.Module):    
    def __init__(self, device):
        super(Biobert_cnn_fc, self).__init__()

        self.model_conf = model_config()
        
        if device == 'cuda2':
            print("Have fun!")
        else:
            
            V = self.model_conf.layer1_features
            D = 768
            C = 9
            Co = 3
            Ks = [self.model_conf.kernel_1,self.model_conf.kernel_2,self.model_conf.kernel_3]
            static = True
            dropout = 0.15
            
            self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
            self.static = static
            self.embed = nn.Embedding(V, D)
            self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
            #self.dropout = nn.Dropout(dropout)
            self.dropout = nn.Dropout(p=0.15)
            self.fc1 = nn.Linear(len(Ks) * Co, C)
            self.softmax = nn.Softmax(dim=1)
            
    def forward(self, ids, segment_ids, mask):
        
        sequence_output, pooled_output = self.bert(
                                                     ids,
                                                     token_type_ids  = segment_ids,
                                                     attention_mask=mask)

        x = sequence_output
        
        if self.static:
            x = Variable(x)
            
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        print("x shape", x.shape)
        print("x type", x.type)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        print("x dropout shape", x.shape)
        print("x dropout type", x.type)
        logit = self.fc1(x)  # (N, C)
        print("logit shape", logit.shape)
        print("logit type", logit.type)
        output = self.softmax(logit)
        print("output shape", output.shape)
        print("output type", output.type)
        return logit



# Implement CRF layer on top of BERT
class Biobert_crf(nn.Module):
    
    def __init__(self):
        super(Biobert_fc, self).__init__()

        self.model_conf = model_config()


        self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))
        self.linear1 = nn.DataParallel(nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes))
#         self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
#         self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)
        
    def forward(self, ids, segment_ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               token_type_ids  = segment_ids,
               attention_mask=mask)
 
          linear1_output  = self.linear1(sequence_output[:,0,:].view(-1,self.model_conf.bert_features)) ## extract the 1st token's embeddings
 
#           linear2_output = self.linear2(linear1_output)
 
          return linear1_output
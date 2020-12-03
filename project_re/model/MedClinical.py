import torch.nn as nn
import math
from transformers import BertModel
import torch
# from torchcrf import CRF
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import LayerNorm
from .lsoftmax import LSoftmaxLinear as Lsoftmax

class Biobert_fc(nn.Module):
    
    def custom_softmax(self, x):
#         print('In softmax function',x)
        y = x.cpu().detach().numpy()
        x = np.array([ [x1[7],x1[1],x1[2],x1[3],x1[4],x1[5],x1[6],x1[0],x1[8]] \
                                   if ( (np.argmax(x1) == 0) and ( x1[0] - x1[7] <  self.model_conf.cust_sftmx_class_beta)) else x1 for x1 in y])

        means = np.mean(x, 1, keepdims=True)[0]
        x_exp = np.exp(x-means)
        x_exp_sum = np.sum(x_exp, 1, keepdims=True)
        final_value = torch.tensor(x_exp/x_exp_sum )
        return torch.tensor(final_value, requires_grad=True)
    
    def __init__(self, device, model_config):
        super(Biobert_fc, self).__init__()

        self.model_conf = model_config
      
        if device == 'cuda':  
            self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))
            self.linear1 = nn.DataParallel(nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes))
            self.dropout = nn.DataParallel(nn.Dropout(model_config.dropout))
            if self.model_conf.act_function =="lsoftmax":
                self.lsoftmax = nn.DataParallel( Lsoftmax(self.model_conf.label_classes, self.model_conf.label_classes, margin=4, device = device))
        else:
            self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
            self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)
            self.dropout = nn.Dropout(model_config.dropout)
            if self.model_conf.act_function =="lsoftmax":
                self.lsoftmax = Lsoftmax(self.model_conf.label_classes, self.model_conf.label_classes, margin=4, device = device)
        
    def forward(self, ids, segment_ids, mask):
        sequence_output, pooled_output = self.bert(
           ids, 
           token_type_ids  = segment_ids,
           attention_mask=mask)

        linear1_output  = self.linear1(sequence_output[:,0,:].view(-1,self.model_conf.bert_features)) ## extract the 1st token's embeddings
        linear1_output = self.dropout(linear1_output)
        if self.model_conf.act_function =="softmax":
            out = torch.softmax(linear1_output, dim=1)
        elif self.model_conf.act_function =="cust_softmax":
            out = self.custom_softmax(linear1_output)
        elif self.model_conf.act_function =="lsoftmax":
            out = self.lsoftmax(linear1_output)
        return (out.squeeze())

   
    
    
class Biobert_cnn_fc(nn.Module):    
    def __init__(self, device, model_config):
        super(Biobert_cnn_fc, self).__init__()

        self.model_conf = model_config
    
#         print('from model', self.model_conf.__dict__, 'in feature', self.model_conf.in_features_fc())

        self.device = device
        if device == 'cuda':
            self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))


            #self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)

            # Convolution layers definition
            self.conv_1 = nn.DataParallel(nn.Conv1d(self.model_conf.bert_features, self.model_conf.out_size, self.model_conf.kernel_1, self.model_conf.stride))
            self.conv_2 = nn.DataParallel(nn.Conv1d(self.model_conf.bert_features, self.model_conf.out_size, self.model_conf.kernel_2, self.model_conf.stride))
            self.conv_3 = nn.DataParallel(nn.Conv1d(self.model_conf.bert_features, self.model_conf.out_size, self.model_conf.kernel_3, self.model_conf.stride))

            # Max pooling layers definition
            self.pool_1 = nn.DataParallel(nn.MaxPool1d(self.model_conf.kernel_1, self.model_conf.stride))
            self.pool_2 = nn.DataParallel(nn.MaxPool1d(self.model_conf.kernel_2, self.model_conf.stride))
            self.pool_3 = nn.DataParallel(nn.MaxPool1d(self.model_conf.kernel_3, self.model_conf.stride))

            self.layer_norm = nn.DataParallel(LayerNorm(self.model_conf.in_features_fc()))
 
            # Fully connected layer definition
            #print("in_features_fc()", self.model_conf.in_features_fc())
            self.fc = nn.DataParallel(nn.Linear(self.model_conf.in_features_fc(), self.model_conf.label_classes))
            if self.model_conf.act_function =="lsoftmax":
                self.lsoftmax =  nn.DataParallel(Lsoftmax(self.model_conf.label_classes, self.model_conf.label_classes, margin=4, device = device))
            
        else:
            self.bert = BertModel.from_pretrained("gsarti/biobert-nli")


            #self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)

            # Convolution layers definition
            self.conv_1 = nn.Conv1d(self.model_conf.bert_features, self.model_conf.out_size, self.model_conf.kernel_1, self.model_conf.stride)
            self.conv_2 = nn.Conv1d(self.model_conf.bert_features, self.model_conf.out_size, self.model_conf.kernel_2, self.model_conf.stride)
            self.conv_3 = nn.Conv1d(self.model_conf.bert_features, self.model_conf.out_size, self.model_conf.kernel_3, self.model_conf.stride)

            # Max pooling layers definition
            self.pool_1 = nn.MaxPool1d(self.model_conf.kernel_1, self.model_conf.stride)
            self.pool_2 = nn.MaxPool1d(self.model_conf.kernel_2, self.model_conf.stride)
            self.pool_3 = nn.MaxPool1d(self.model_conf.kernel_3, self.model_conf.stride)

            self.layer_norm = LayerNorm(self.model_conf.in_features_fc())
            
            # Fully connected layer definition
            #print("in_features_fc()", self.model_conf.in_features_fc())
            self.fc = nn.Linear(self.model_conf.in_features_fc(), self.model_conf.label_classes)
            if self.model_conf.act_function =="lsoftmax":
                self.lsoftmax = Lsoftmax(self.model_conf.label_classes, self.model_conf.label_classes, margin=4, device = device)
            
        #7

    def custom_softmax(self, x):
#         print('In softmax function',x)
        y = x.cpu().detach().numpy()
        x = np.array([ [x1[7],x1[1],x1[2],x1[3],x1[4],x1[5],x1[6],x1[0],x1[8]] \
                                   if ( (np.argmax(x1) == 0) and ( x1[0] - x1[7] < 0.005)) else x1 for x1 in y])

        means = np.mean(x, 1, keepdims=True)[0]
        x_exp = np.exp(x-means)
        x_exp_sum = np.sum(x_exp, 1, keepdims=True)
        final_value = torch.tensor(x_exp/x_exp_sum )
        return torch.tensor(final_value, requires_grad=True, device = self.device)
        

    def forward(self, ids, segment_ids, mask):

        sequence_output, pooled_output = self.bert(
                                                 ids,
                                                 token_type_ids  = segment_ids,
                                                 attention_mask=mask)

        #print("sequence output type: ", type(sequence_output))
        #           print("sequence output shape: ", sequence_output.shape)
        x = sequence_output
        x = torch.transpose(x,1,2)
        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        #print("x1 shape: ", x1.shape)

        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)

        #print("x2 shape: ", x2.shape)

        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        #           x3 = self.layer_norm(x3)

        #print("x3 shape: ", x3.shape)

        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3), 2)
        #print("union  type: ", type(union))
        #print("union  shape: ", union.shape)

        union = union.reshape(union.size(0), -1)
        #print("union reshape  type: ", type(union))
        #print("union reshape  shape: ", union.shape)

        union = self.layer_norm(union)
        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)
        #print("out shape pre softmax", out.shape)
        # Activation function is applied
        if self.model_conf.act_function =="softmax":
            out = torch.softmax(out, dim=1)
        elif self.model_conf.act_function =="cust_softmax":
            out = self.custom_softmax(out)
        elif self.model_conf.act_function =="lsoftmax":
            out = self.lsoftmax(out)
            
        return (out.squeeze())
        #print("out shape", out.squeeze().shape)


        



# # Implement CRF layer on top of BERT
# class Biobert_crf(nn.Module):
    
#     def __init__(self, device):
#         start_label_id =0
#         stop_label_id = 8
#         super(Biobert_crf, self).__init__()

#         self.model_conf = model_config()

#         if device == 'cuda2':
#             self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))
#             self.dropout = nn.DataParallel(torch.nn.Dropout(0.2))
#             # Maps the output of the bert into label space.
#             self.hidden2label = nn.DataParallel(nn.Linear(self.model_conf.bert_features,  self.model_conf.label_classes))
#             self.crf =  nn.DataParallel(CRF(self.model_conf.label_classes))
#         else:
#             self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
#             self.dropout = torch.nn.Dropout(0.2)
#             # Maps the output of the bert into label space.
#             self.hidden2label = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)
#             self.crf = CRF(self.model_conf.label_classes)
            
# #         self.transitions = nn.Parameter(torch.randn(self.model_conf.label_classes, self.model_conf.label_classes))
# #         self.transitions.data[start_label_id, :] = -10000
# #         self.transitions.data[:, stop_label_id] = -10000

#     def forward(self, ids, segment_ids, mask):
# #         tags =  torch.tensor([
# #             [0, 1], [2, 4], [3, 1],[7,8]], dtype=torch.long)  # (seq_length, batch_size)
#         sequence_output, pooled_output = self.bert(
#                ids, 
#                token_type_ids  = segment_ids,
#                attention_mask=mask)
#         bert_seq_out = self.dropout(sequence_output)
#         bert_seq_out = self.dropout(bert_seq_out)
#         bert_feats = self.hidden2label(bert_seq_out)
#         print(bert_feats.shape)
#         x1 = self.crf.decode(bert_feats)
# #         print('shape:', np.array(x1).shape)
# #         print(x1)
#         return torch.Tensor(x1)
        
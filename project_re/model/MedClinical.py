import torch.nn as nn
import math
from transformers import BertModel
import torch
from .model_config import model_config 

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
    
        print('from model', self.model_conf.__dict__, 'in feature', self.model_conf.in_features_fc())
        
        if device == 'cuda2':
            self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))


            #self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)

            # Convolution layers definition
            self.conv_1 = nn.DataParallel(nn.Conv1d(self.model_conf.layer1_features, self.model_conf.out_size, self.model_conf.kernel_1, self.model_conf.stride))
            self.conv_2 = nn.DataParallel(nn.Conv1d(self.model_conf.layer1_features, self.model_conf.out_size, self.model_conf.kernel_2, self.model_conf.stride))
            self.conv_3 = nn.DataParallel(nn.Conv1d(self.model_conf.layer1_features, self.model_conf.out_size, self.model_conf.kernel_3, self.model_conf.stride))

            # Max pooling layers definition
            self.pool_1 = nn.DataParallel(nn.MaxPool1d(self.model_conf.kernel_1, self.model_conf.stride))
            self.pool_2 = nn.DataParallel(nn.MaxPool1d(self.model_conf.kernel_2, self.model_conf.stride))
            self.pool_3 = nn.DataParallel(nn.MaxPool1d(self.model_conf.kernel_3, self.model_conf.stride))


            # Fully connected layer definition
            #print("in_features_fc()", self.model_conf.in_features_fc())
            self.fc = nn.DataParallel(nn.Linear(self.model_conf.in_features_fc(), self.model_conf.label_classes))
        else:
            self.bert = BertModel.from_pretrained("gsarti/biobert-nli")


            #self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)

            # Convolution layers definition
            self.conv_1 = nn.Conv1d(self.model_conf.layer1_features, self.model_conf.out_size, self.model_conf.kernel_1, self.model_conf.stride)
            self.conv_2 = nn.Conv1d(self.model_conf.layer1_features, self.model_conf.out_size, self.model_conf.kernel_2, self.model_conf.stride)
            self.conv_3 = nn.Conv1d(self.model_conf.layer1_features, self.model_conf.out_size, self.model_conf.kernel_3, self.model_conf.stride)

            # Max pooling layers definition
            self.pool_1 = nn.MaxPool1d(self.model_conf.kernel_1, self.model_conf.stride)
            self.pool_2 = nn.MaxPool1d(self.model_conf.kernel_2, self.model_conf.stride)
            self.pool_3 = nn.MaxPool1d(self.model_conf.kernel_3, self.model_conf.stride)


            # Fully connected layer definition
            #print("in_features_fc()", self.model_conf.in_features_fc())
            self.fc = nn.Linear(self.model_conf.in_features_fc(), self.model_conf.label_classes)
            
        #7


    def forward(self, ids, segment_ids, mask):

          sequence_output, pooled_output = self.bert(
                                                     ids,
                                                     token_type_ids  = segment_ids,
                                                     attention_mask=mask)

          #print("sequence output type: ", type(sequence_output))
          #print("sequence output type: ", sequence_output.shape)
          x = sequence_output
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

          #print("x3 shape: ", x3.shape)

          # The output of each convolutional layer is concatenated into a unique vector
          union = torch.cat((x1, x2, x3), 2)
          #print("union  type: ", type(union))
          #print("union  shape: ", union.shape)

          union = union.reshape(union.size(0), -1)
         #print("union reshape  type: ", type(union))
          #print("union reshape  shape: ", union.shape)

          # The "flattened" vector is passed through a fully connected layer
          out = self.fc(union)
          #print("out shape pre softmax", out.shape)
          # Activation function is applied
          out = torch.softmax(out, dim=1)
          #print("out shape", out.squeeze().shape)

          return out.squeeze()


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
        
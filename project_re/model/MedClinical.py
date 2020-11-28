import torch.nn as nn
import math
from transformers import BertModel
import torch
from torchcrf import CRF
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Biobert_fc(nn.Module):
    
    def __init__(self, device, model_config):
        super(Biobert_fc, self).__init__()

        self.model_conf = model_config
      
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
    def __init__(self, device, model_config):
        super(Biobert_cnn_fc, self).__init__()

        self.model_conf = model_config
    
        print('from model', self.model_conf.__dict__, 'in feature', self.model_conf.in_features_fc())
        
        if device == 'cuda2':
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


            # Fully connected layer definition
            #print("in_features_fc()", self.model_conf.in_features_fc())
            self.fc = nn.DataParallel(nn.Linear(self.model_conf.in_features_fc(), self.model_conf.label_classes))
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

        
class Biobert_cnn_fc_new(nn.Module):    
    
    def __init__(self, device, model_config):
        super(Biobert_cnn_fc_new, self).__init__()

        self.model_conf = model_config
    
#         print('from model', self.model_conf.__dict__, 'in feature', self.model_conf.in_features_fc())
        kernel_sizes = [24,30,36] #self.model_conf.KERNEL_SIZES
        kernel_1 = [self.model_conf.kernel_1,self.model_conf.bert_features ]
        kernel_2 = [self.model_conf.kernel_2,self.model_conf.bert_features ]
        kernel_3 = [self.model_conf.kernel_3,self.model_conf.bert_features ]
        
        kerenl_num = 3
        embeding_dimension = self.model_conf.bert_features
        
        if device == 'cuda2':
            self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))


            #self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)

            # Convolution layers definition
            self.conv_1 = nn.DataParallel(nn.Conv2d(self.model_conf.layer1_features, self.model_conf.out_size, kernel_1, self.model_conf.stride))
            self.conv_2 = nn.DataParallel(nn.Conv2d(self.model_conf.layer1_features, self.model_conf.out_size, kernel_2, self.model_conf.stride))
            self.conv_3 = nn.DataParallel(nn.Conv2d(self.model_conf.layer1_features, self.model_conf.out_size, kernel_3, self.model_conf.stride))

            # Max pooling layers definition
            self.pool_1 = nn.DataParallel(nn.MaxPool1d(kernel_1, self.model_conf.stride))
            self.pool_2 = nn.DataParallel(nn.MaxPool1d(kernel_2, self.model_conf.stride))
            self.pool_3 = nn.DataParallel(nn.MaxPool1d(kernel_3, self.model_conf.stride))


            # Fully connected layer definition
            #print("in_features_fc()", self.model_conf.in_features_fc())
            self.fc = nn.DataParallel(nn.Linear(self.model_conf.in_features_fc(), self.model_conf.label_classes))
        else:
            self.bert = BertModel.from_pretrained("gsarti/biobert-nli")


            #self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)

            # Convolution layers definition
            self.conv_1 = nn.Conv2d(1, 3, (24,768), (2,1))
            self.conv_2 = nn.Conv2d(1,3 , (36,768), (2,1))
            self.conv_3 = nn.Conv2d(1,3, (48,768), (2,1))

            # Max pooling layers definition
            self.pool_1 = nn.MaxPool2d((24,768), (2,1))
            self.pool_2 = nn.MaxPool2d((36,768), (2,1))
            self.pool_3 = nn.MaxPool2d((48,768), (2,1))


            # Fully connected layer definition
            #print("in_features_fc()", self.model_conf.in_features_fc())
            self.fc = nn.Linear(9, self.model_conf.label_classes)
            

#  Different implementation            
#             self.bert = BertModel.from_pretrained("gsarti/biobert-nli")


#             #self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)

#             # Convolution layers definition
#             self.convs1 = nn.ModuleList([nn.Conv2d(1, kerenl_num, (K, embeding_dimension)) for K in kernel_sizes])
# #             self.dropout = nn.Dropout(0.1)
#             self.fc = nn.Linear(len(kernel_sizes) * kerenl_num, self.model_conf.label_classes)
        



        
    def forward(self, ids, segment_ids, mask):

          sequence_output, pooled_output = self.bert(
                                                     ids,
                                                     token_type_ids  = segment_ids,
                                                     attention_mask=mask)

          #print("sequence output type: ", type(sequence_output))
          print("sequence output type: ", sequence_output.shape)
          x = sequence_output
        
          x = x.unsqueeze(1)
          print('shape x1', x.shape) 
          # Convolution layer 1 is applied
          x1 = self.conv_1(x)
          print('shape x1', x1.shape)
          x1 = torch.relu(x1.squeeze(3))
          print('shape x1', x1.shape)
          x1 = self.pool_1(x1.squeeze(2))
            

          #print("x1 shape: ", x1.shape)

          # Convolution layer 2 is applied
          x2 = self.conv_2(x)
          x2 = torch.relu((x2.squeeze(3)))
          x2 = self.pool_2(x2.squeeze(2))

          #print("x2 shape: ", x2.shape)

          # Convolution layer 3 is applied
          x3 = self.conv_3(x)
          x3 = torch.relu(x3.squeeze(3))
          x3 = self.pool_3(x3.squeeze(2))

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
        
# Different implementation

#           #print("sequence output type: ", type(sequence_output))
#           #print("sequence output type: ", sequence_output.shape)
#         x = sequence_output
# #         if self.static:
#         x = Variable(x)

#         x = x.unsqueeze(1)  # (N, Ci, W, D)
#         x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
#         x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
#         union = torch.cat(x, 1)
# #         x = self.dropout(x)  # (N, len(Ks)*Co)
#         union = union.reshape(union.size(0), -1)
    
#         logit = self.fc(union)  # (N, C)

#         out = torch.softmax(logit, dim=1)

#         return out.squeeze()



class ConvNet(nn.Module):
    
    def __init__(self, device, model_config):
        super(ConvNet, self).__init__()
        in_size = 256
        hid1_size = 16
        hid2_size = 32
        k_conv_size = 5
        out_size = 9
        
        self.model_conf = model_config
        self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_size, hid1_size, k_conv_size ),
            nn.BatchNorm1d(hid1_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
        
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(hid1_size, hid2_size, k_conv_size),
#             nn.BatchNorm2d(hid2_size),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2))

#           out_conv_1 = ((self.in_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
#           #767
#           #((768 - 1*(2-1) -1)/2) + 1 = 384
#           out_conv_1 = math.floor(out_conv_1)
#           #384
#           out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
#           #((384 - 1 * (2 - 1) - 1) / 2) + 1 = 192
#           #766
#           out_pool_1 = math.floor(out_pool_1)
            
        out_pool_1 = ((hid1_size - 1 * (k_conv_size - 1) - 1) / 1) + 1
        self.fc = nn.Linear(out_pool_1 , out_size)
        
    def forward(self,  ids, segment_ids, mask):
        
        sequence_output, pooled_output = self.bert(
                                                 ids,
                                                 token_type_ids  = segment_ids,
                                                 attention_mask=mask)

        #print("sequence output type: ", type(sequence_output))
        print("sequence output type: ", sequence_output.shape)
        x = sequence_output

#         x = x.unsqueeze(1)    
#         x = torch.transpose(x,)
        out = self.layer1(x)
#         out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = torch.softmax(out, dim=1)
        return out
    


# Implement CRF layer on top of BERT
class Biobert_crf(nn.Module):
    
    def __init__(self, device):
        start_label_id =0
        stop_label_id = 8
        super(Biobert_crf, self).__init__()

        self.model_conf = model_config()

        if device == 'cuda2':
            self.bert = nn.DataParallel(BertModel.from_pretrained("gsarti/biobert-nli"))
            self.dropout = nn.DataParallel(torch.nn.Dropout(0.2))
            # Maps the output of the bert into label space.
            self.hidden2label = nn.DataParallel(nn.Linear(self.model_conf.bert_features,  self.model_conf.label_classes))
            self.crf =  nn.DataParallel(CRF(self.model_conf.label_classes))
        else:
            self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
            self.dropout = torch.nn.Dropout(0.2)
            # Maps the output of the bert into label space.
            self.hidden2label = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)
            self.crf = CRF(self.model_conf.label_classes)
            
#         self.transitions = nn.Parameter(torch.randn(self.model_conf.label_classes, self.model_conf.label_classes))
#         self.transitions.data[start_label_id, :] = -10000
#         self.transitions.data[:, stop_label_id] = -10000

    def forward(self, ids, segment_ids, mask):
#         tags =  torch.tensor([
#             [0, 1], [2, 4], [3, 1],[7,8]], dtype=torch.long)  # (seq_length, batch_size)
        sequence_output, pooled_output = self.bert(
               ids, 
               token_type_ids  = segment_ids,
               attention_mask=mask)
        bert_seq_out = self.dropout(sequence_output)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        print(bert_feats.shape)
        x1 = self.crf.decode(bert_feats)
#         print('shape:', np.array(x1).shape)
#         print(x1)
        return torch.Tensor(x1)
        
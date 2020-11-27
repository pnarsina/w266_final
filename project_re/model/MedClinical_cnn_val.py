import torch.nn as nn
from transformers import BertModel
import math
import torch
# from .model_config import model_config

class model_config:
    def __init__(self):
        self.bert_features = 768
        self.layer1_features = 256

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 8
        self.kernel_2 = 10
        self.kernel_3 = 12

        # Output size for each convolution (number of convolution channel)
        self.out_size = 768

        # Number of strides for each convolution
        self.stride = 2

# For TACRED data
#         self.label_classes = 42
# For i2b2
        self.label_classes = 9

class Biobert_fc(nn.Module):
    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(Biobert_fc, self).__init__()

        self.model_conf = model_config()
        self.bert = BertModel.from_pretrained("gsarti/biobert-nli")

        V = self.model_conf.layer1_features
        D = self.model_conf.bert_features
        C = self.model_conf.label_classes
        Co = 3 #kernel_num
        Ks = [2,3,4]

        self.static = static
        #self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()


    def forward(self, ids, segment_ids, mask):

        sequence_output, pooled_output = self.bert(
                                                     ids,
                                                     token_type_ids  = segment_ids,
                                                     attention_mask=mask)

          #print("sequence output type: ", type(sequence_output))
          #print("sequence output type: ", sequence_output.shape)
        x = sequence_output
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.softmax(logit)
        return output

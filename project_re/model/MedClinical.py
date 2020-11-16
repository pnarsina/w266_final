import torch.nn as nn
from transformers import BertModel
# from .model_config import model_config 

class model_config:
    def __init__(self):
        self.bert_features = 768
        self.layer1_features = 256
# For TACRED data
#         self.label_classes = 42  
# For i2b2
        self.label_classes = 9
    
class Biobert_fc(nn.Module):
    
    def __init__(self):
        super(Biobert_fc, self).__init__()

        self.model_conf = model_config()

        self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
        self.linear1 = nn.Linear(self.model_conf.bert_features, self.model_conf.label_classes)
#         self.linear2 = nn.Linear(self.model_conf.layer1_features, self.model_conf.label_classes) ## 3 is the number of classes in this example
        
    def forward(self, ids, segment_ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               token_type_ids  = segment_ids,
               attention_mask=mask)
 
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,self.model_conf.bert_features)) ## extract the 1st token's embeddings
 
#           linear2_output = self.linear2(linear1_output)
 
          return linear1_output
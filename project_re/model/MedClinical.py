import torch.nn as nn
from transformers import BertModel
from config import model_config 

class Biobert_fc(nn.Module):
    
    def __init__(self):
        super(Biobert_fc, self).__init__()
        
        self.bert = BertModel.from_pretrained("gsarti/biobert-nli")
        self.linear1 = nn.Linear(model_config.bert_features, model_config.layer1_features)
        self.linear2 = nn.Linear(model_config.layer1_features, model_config.label_classes) ## 3 is the number of classes in this example
        
    def forward(self, ids, segment_ids, mask):
          sequence_output, pooled_output = self.bert(
               ids, 
               token_type_ids  = segment_ids,
               attention_mask=mask)
 
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,model_config.bert_features)) ## extract the 1st token's embeddings
 
          linear2_output = self.linear2(linear1_output)
 
          return linear2_output
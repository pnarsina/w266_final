import csv
import os
import sys
import logging
import json, math
from json import JSONEncoder

logger = logging.getLogger()

class config(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [config(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, config(b) if isinstance(b, dict) else b)

            
class configEncoder(JSONEncoder):

    def default(self, object):
        if isinstance(object, config):
            return object.__dict__
        else:
            return json.JSONEncoder.default(self, object)

        
def load_config(config_folder):
            
    with open(os.path.join(config_folder, "config.json")) as f:
        config1 = config(json.load(f))
    return config1


def load_model_config(config_folder):
    config1 = load_config(config_folder)
    return config1.modelconfig     


class model_config:
    def __init__(self):
        mod_config = load_model_config("config")
        self.config = mod_config
        self.bert_features = mod_config.BERT_FEATURES
        self.layer1_features = mod_config.LAYER1_FEATURES
        self.max_seq_length = 256
        self.label_classes = mod_config.LBAEL_CLASSES

    # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 24 #mod_config.KERNEL_1
        self.kernel_2 = 36 #mod_config.KERNEL_2
        self.kernel_3 = 48 #mod_config.KERNEL_3

        # Output size for each convolution (number of convolution channel)
        self.out_size = mod_config.OUT_SIZE

        # Number of strides for each convolution
        self.stride = mod_config.STRIDE
        self.dropout = mod_config.DROP_OUT
        self.act_function = mod_config.ACT_FUNCTION
        self.cust_sftmx_class_beta = mod_config.CUST_SFTMX_CLASS_BETA
        
    def in_features_fc(self):
          '''Calculates the number of output features after Convolution + Max pooling

          Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
          Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

          source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
          '''
          # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
#           print('Inside in_features:', self.bert_features, self.kernel_1,self.kernel_2,self.kernel_3,self.out_size)
          out_conv_1 = ((self.max_seq_length - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
          #767
          #((768 - 1*(2-1) -1)/2) + 1 = 384
          out_conv_1 = math.floor(out_conv_1)
          #384
          out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
          #((384 - 1 * (2 - 1) - 1) / 2) + 1 = 192
          #766
          out_pool_1 = math.floor(out_pool_1)
          #192

          # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
          out_conv_2 = ((self.max_seq_length - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
          #((768 - 1*(4-1) -1)/2) + 1 = 383
          out_conv_2 = math.floor(out_conv_2)
          #383
          out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
          #((383 - 1 * (4 - 1) - 1) / 2) + 1 = 190.5
          out_pool_2 = math.floor(out_pool_2)
          #190

          # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
          out_conv_3 = ((self.max_seq_length - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
          #((768 - 1*(6-1) -1)/2) + 1 = 382
          out_conv_3 = math.floor(out_conv_3)
          #382
          out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
          #((382 - 1*(6-1) - 1)/2) + 1 = 189
          out_pool_3 = math.floor(out_pool_3)
          #189

          # Returns "flattened" vector (input for fully connected layer)
          return (out_pool_1 + out_pool_2 + out_pool_3) * self.out_size

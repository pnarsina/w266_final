import csv
import os
import sys
import logging
import json
import config
from json import JSONEncoder
from pathlib import Path, PureWindowsPath, PurePosixPath
import math

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
        
def load_model_config(config_folder):
    curr_dir = Path(config_folder  )
    file_path = curr_dir.parent / config_folder /'config.json'
    with file_path.open() as f:
        config1 = config(json.load(f))
#     print("retrieved configuraiton", config1.modelconfig.__dict__)
    return config1.modelconfig     


class model_config:
    def __init__(self):
        mod_config = load_model_config("config")
        self.config = mod_config
        self.bert_features = mod_config.BERT_FEATURES
        self.layer1_features = mod_config.LAYER1_FEATURES

        self.label_classes = mod_config.LBAEL_CLASSES

    # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = mod_config.KERNEL_1
        self.kernel_2 = mod_config.KERNEL_2
        self.kernel_3 = mod_config.KERNEL_3

        # Output size for each convolution (number of convolution channel)
        self.out_size = mod_config.OUT_SIZE

        # Number of strides for each convolution
        self.stride = mod_config.STRIDE
        
    def in_features_fc(self):
          '''Calculates the number of output features after Convolution + Max pooling

          Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
          Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

          source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
          '''
          # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
#           print('Inside in_features:', self.bert_features, self.kernel_1,self.kernel_2,self.kernel_3,self.out_size)
          out_conv_1 = ((self.bert_features - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
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
          out_conv_2 = ((self.bert_features - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
          #((768 - 1*(4-1) -1)/2) + 1 = 383
          out_conv_2 = math.floor(out_conv_2)
          #383
          out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
          #((383 - 1 * (4 - 1) - 1) / 2) + 1 = 190.5
          out_pool_2 = math.floor(out_pool_2)
          #190

          # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
          out_conv_3 = ((self.bert_features - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
          #((768 - 1*(6-1) -1)/2) + 1 = 382
          out_conv_3 = math.floor(out_conv_3)
          #382
          out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
          #((382 - 1*(6-1) - 1)/2) + 1 = 189
          out_pool_3 = math.floor(out_pool_3)
          #189

          # Returns "flattened" vector (input for fully connected layer)
          return (out_pool_1 + out_pool_2 + out_pool_3) * self.out_size

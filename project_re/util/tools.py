import csv
import os
import sys
import logging
import json
import config

logger = logging.getLogger()

class config(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [config(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, config(b) if isinstance(b, dict) else b)

def load_config(config_folder):
            
    with open(os.path.join(config_folder, "config.json")) as f:
        config1 = config(json.load(f))
    return config1
import csv
import os
import sys
import logging
import json
import config


def load_config(config_folder):
    class obj(object):
        def __init__(self, d):
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                   setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
                else:
                   setattr(self, a, obj(b) if isinstance(b, dict) else b)
            
    with open(os.path.join(config_folder, "config.json")) as f:
        config = obj(json.load(f))
    return config
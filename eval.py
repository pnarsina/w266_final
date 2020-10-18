"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
import config

def evaluate_model(evalparams):



    torch.manual_seed(evalparams.seed)
    random.seed(1234)
    if evalparams.cpu:
        evalparams.cuda = False
    elif evalparams.cud:
        torch.cuda.manual_seed(args.seed)
        
    # load opt
    print(evalparams.model_dir, evalparams.model)
#     model_file = evalparams.model_dir + "/" + evalparams.model
    model_file = 'best_model.pt'
    print("Loading model from {}".format(model_file))
    opt = torch_utils.load_config(model_file)
    model = RelationModel(opt)
    model.load(model_file)

    # load vocab
    vocab_file = evalparams.model_dir + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

    # load data
    data_file = opt['data_dir'] + '/{}.json'.format(evalparams.dataset)
    print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
    batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

    helper.print_config(opt)
    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

    predictions = []
    all_probs = []
    for i, b in enumerate(batch):
        preds, probs, _ = model.predict(b)
        predictions += preds
        all_probs += probs
    predictions = [id2label[p] for p in predictions]
    p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

    # save probability scores
    if len(evalparams.out) > 0:
        helper.ensure_dir(os.path.dirname(evalparams.out))
        with open(evalparams.out, 'wb') as outfile:
            pickle.dump(all_probs, outfile)
        print("Prediction scores saved to {}.".format(evalparams.out))

    print("Evaluation ended.")



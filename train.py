"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import scorer, constant, helper
from utils.vocab import Vocab

import config



def train_model(vocab_params, train_params, train_batch, dev_batch, model_id=-1):
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    random.seed(train_params.seed)
    
    if train_params.cpu:
        train_params.cuda = False
    elif train_params.cuda:
        torch.cuda.manual_seed(train_params.seed)

    # make opt
    opt = vars(vocab_params)
    
    print(constant.LABEL_TO_ID)
    print(opt)
    opt['num_class'] = len(constant.LABEL_TO_ID)
#     Combine all the parameters together
    opt.update(vars(train_params))
    
    # load vocab
    vocab_file = opt['vocab_dir'] + '/vocab.pkl'
    vocab = Vocab(vocab_file, load=True)
    opt['vocab_size'] = vocab.size
    emb_file = opt['vocab_dir'] + '/embedding.npy'
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == vocab.size
    assert emb_matrix.shape[1] == opt['emb_dim']

    if(model_id ==-1):
        model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + model_id
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)

    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
    vocab.save(model_save_dir + '/vocab.pkl')
    file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

    # print model info
    helper.print_config(opt)

    # model
    model = RelationModel(opt, emb_matrix=emb_matrix)

    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
    dev_f1_history = []
    current_lr = opt['lr']

    global_step = 0
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_batch) * opt['num_epoch']

    # start training
    for epoch in range(1, opt['num_epoch']+1):
        train_loss = 0
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = model.update(batch)
            train_loss += loss
            if global_step % opt['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                        opt['num_epoch'], loss, duration, current_lr))

        # eval on dev
        print("Evaluating on dev set...")
        predictions = []
        dev_loss = 0
        for i, batch in enumerate(dev_batch):
            preds, _, loss = model.predict(batch)
            predictions += preds
            dev_loss += loss
        predictions = [id2label[p] for p in predictions]
        dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)

        train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
        dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
        print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
                train_loss, dev_loss, dev_f1))
        file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

        # save
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        model.save(model_file, epoch)
        if epoch == 1 or dev_f1 > max(dev_f1_history):
            copyfile(model_file, model_save_dir + '/best_model.pt')
            print("new best model saved.")
        if epoch % opt['save_epoch'] != 0:
            os.remove(model_file)

        # lr schedule
        if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
                opt['optim'] in ['sgd', 'adagrad']:
            current_lr *= opt['lr_decay']
            model.update_lr(current_lr)

        dev_f1_history += [dev_f1]
        print("")

    print("Training ended with {} epochs.".format(epoch))


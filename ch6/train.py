import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import sys
from MY_BiLSTM_Model import MY_BiLSTM_Model
from MY_BiLSTMCRF_Model import MY_BiLSTMCRF_Model
from dataset import CnNewsDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bilstm_train():
    ''' main function for trainning
    '''
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    BATCH_SIZE = 64
    EPOCH_SIZE = 100
    cnd = CnNewsDataset(fname='199801.txt',
                        is_preprocess=True, batch_size=BATCH_SIZE)
    VOCAB_SIZE = len(cnd.text_field.vocab)
    tag_to_ix = cnd.label_field.vocab.stoi
    
    NUM_LABELS = len(tag_to_ix)
    ix_to_tag = dict(zip(range(NUM_LABELS), cnd.label_field.vocab.itos))
    model = MY_BiLSTM_Model(NUM_LABELS, BATCH_SIZE,
                            HIDDEN_DIM, VOCAB_SIZE, EMBEDDING_DIM)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i in range(EPOCH_SIZE):
        train_iter, dev_iter, test_iter = cnd.get_data_iter()
        print('epoch size', i)
        loss_sum = torch.tensor([0.])
        for j, batch in enumerate(train_iter):
            if j != 0 and j % 10 == 0:
                print('loss average:', loss_sum / 10)
                loss_sum = torch.tensor([0.])
            feature, target = batch.text, batch.label
            feature = feature.to(device)
            target = target.to(device)
            feature.t_(), target.t_()
            mask = feature != tag_to_ix['<pad>']  # + (feature == 3)
            model.zero_grad()
            loss = model(feature, mask, target)
            loss_sum += loss
            loss.backward()
            optimizer.step()

def bilstm_crf_train():
    ''' main function for trainning
    '''
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    BATCH_SIZE = 64
    EPOCH_SIZE = 100
    cnd = CnNewsDataset(fname='199801.txt',
                        is_preprocess=True, batch_size=BATCH_SIZE)
    VOCAB_SIZE = len(cnd.text_field.vocab)
    tag_to_ix = cnd.label_field.vocab.stoi
    tag_to_ix['<sos>'] = len(tag_to_ix)
    NUM_LABELS = len(tag_to_ix)
    ix_to_tag = dict(zip(range(NUM_LABELS), cnd.label_field.vocab.itos))
    model = MY_BiLSTMCRF_Model (NUM_LABELS, BATCH_SIZE,
                            HIDDEN_DIM, VOCAB_SIZE, EMBEDDING_DIM)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i in range(EPOCH_SIZE):
        train_iter, dev_iter, test_iter = cnd.get_data_iter()
        print('epoch size', i)
        loss_sum = torch.tensor([0.])
        for j, batch in enumerate(train_iter):
            if j != 0 and j % 10 == 0:
                print('loss average:', loss_sum / 10)
                loss_sum = torch.tensor([0.])
            feature, target = batch.text, batch.label
            feature = feature.to(device)
            target = target.to(device)
            feature.t_(), target.t_()
            mask = feature != tag_to_ix['<pad>']  # + (feature == 3)
            model.zero_grad()
            loss = model(feature, mask, target)
            loss_sum += loss
            loss.backward()
            optimizer.step()








"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/11/10 10:20
"""

import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
import sys
import json
from BERT.BERT_Torch.BERT_Model import BERT
from BERT.BERT_Torch.BERT_Prepro import data_loader, convert_features
from BERT.transformers.transformers import AdamW, WarmupLinearSchedule
from BERT.transformers.transformers import BertTokenizer


def train(config):
    logger_name = "my_log_train"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(config.target_dir + "/" + config.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info('loading data...')
    train_loader, train_num = data_loader(config.train_eval,
                                          config.batch_size,
                                          config.max_length_seq,
                                          config.max_length_query,
                                          shuffle=True)
    dev_loader, dev_num = data_loader(config.dev_eval,
                                      config.batch_size,
                                      config.max_length_seq,
                                      config.max_length_query,
                                      shuffle=False)

    logger.info('train size # sent ' + str(len(train_loader)))
    logger.info('dev size # sent ' + str(len(dev_loader)))
    model = BERT.from_pretrained(config.pre_path)
    if config.is_gpu:
        model.cuda()
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config.num_warmup_steps, t_total=config.num_train_steps)

    criterion = nn.CrossEntropyLoss()
    logger.info('start to train...')

    total = 0.
    correct = 0.
    loss_data = 0.
    timer = time.time()
    best_acc = 0.
    early_stop = 0
    for epoch in range(config.epoch):
        for i, (input_ids, input_mask, segment_ids, labels) in enumerate(train_loader):
            if config.is_gpu:
                input_ids_batch, input_mask_batch, segment_ids_batch, labels_batch = Variable(input_ids.cuda()), \
                                                                                    Variable(input_mask.cuda()), \
                                                                                     Variable(segment_ids.cuda()), \
                                                                                    Variable(labels.cuda())
            else:
                input_ids_batch, input_mask_batch, segment_ids_batch, labels_batch = Variable(input_ids), \
                                                                                     Variable(input_mask), \
                                                                                     Variable(segment_ids), \
                                                                                     Variable(labels)
            out = model(input_ids_batch, input_mask_batch, segment_ids_batch)
            optimizer.zero_grad()
            loss = criterion(out, labels_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            _, predict = out.data.max(dim=1)
            total += labels_batch.size(0)
            correct += (torch.sum(predict == labels_batch.data)).cpu().numpy()
            loss_data += (loss.data * labels_batch.size(0))
            if (i + 1) % config.display_interval == 0 or (i + 2) == train_num:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, train-loss %.3f, time %.2fs, ' %
                            (epoch, i + 1, len(input_ids_batch), correct / total, loss_data / total,
                             time.time() - timer))
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
                model.eval()
                dev_correct = 0.
                dev_total = 0.
                dev_loss_data = 0.
                for i_dev, (input_ids, input_mask, segment_ids, labels) in enumerate(dev_loader):
                    if config.is_gpu:
                        dev_input_ids_batch, dev_input_mask_batch, dev_segment_ids_batch, dev_labels_batch = Variable(input_ids.cuda()), \
                                    Variable(input_mask.cuda()), Variable(segment_ids.cuda()), Variable(labels.cuda())
                    else:
                        dev_input_ids_batch, dev_input_mask_batch, dev_segment_ids_batch, dev_labels_batch = Variable(input_ids), \
                                    Variable(input_mask), Variable(segment_ids), Variable(labels)
                    dev_out = model(dev_input_ids_batch, dev_input_mask_batch, dev_segment_ids_batch)
                    dev_loss = criterion(dev_out, dev_labels_batch)
                    _, dev_predict = dev_out.data.max(dim=1)

                    dev_total += dev_labels_batch.size(0)
                    dev_correct += (torch.sum(dev_predict == dev_labels_batch.data)).cpu().numpy()
                    dev_loss_data += (dev_loss.data * dev_labels_batch.size(0))
                dev_acc = dev_correct / dev_total
                logger.info('dev-loss %.3f, dev-acc %.3f' % (dev_loss_data / dev_total, dev_acc))

                if dev_acc > best_acc:
                    torch.save(model.state_dict(), config.save_dir + '/' + 'siamese.pt')
                    logger.info('save model')
                    best_acc = dev_acc
                    early_stop = 0
                else:
                    early_stop = early_stop + 1
                if early_stop > config.early_stop:
                    logger.info('training end!')
                    break
                model.train()
    logger.info('training end!')


def predict(config):
    logger_name = "my_log_predict"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(config.target_dir + "/" + config.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info('loading data...')
    dev_loader, dev_num = data_loader(config.dev_eval,
                                      config.batch_size,
                                      config.max_length_content,
                                      shuffle=False)
    logger.info('dev size # sent ' + str(len(dev_loader)))

    model = BERT.from_pretrained(config.pre_path)
    if config.is_gpu:
        model.cuda()
    model.load_state_dict(torch.load(config.save_dir + '/' +'siamese.pt'))
    model.eval()
    correct = 0.
    total = 0.
    loss_data = 0.
    criterion = nn.CrossEntropyLoss()
    for i, (input_ids, input_mask, segment_ids, labels) in enumerate(dev_loader):
        if config.is_gpu:
            dev_input_ids_batch, dev_input_mask_batch, dev_segment_ids_batch, dev_labels_batch = Variable(input_ids.cuda()), \
                                                                                                 Variable(input_mask.cuda()), \
                                                                                                 Variable(segment_ids.cuda()), \
                                                                                                 Variable(labels.cuda())
        else:
            dev_input_ids_batch, dev_input_mask_batch, dev_segment_ids_batch, dev_labels_batch = Variable(input_ids), \
                                                                                                 Variable(input_mask), \
                                                                                                 Variable(segment_ids), \
                                                                                                 Variable(labels)

        dev_out = model(dev_input_ids_batch, dev_input_mask_batch, dev_segment_ids_batch)
        dev_loss = criterion(dev_out, dev_labels_batch)
        _, dev_predict = dev_out.data.max(dim=1)
        total += dev_labels_batch.size(0)
        correct += (torch.sum(dev_predict == dev_labels_batch.data)).cpu().numpy()
        loss_data += (dev_loss.data * dev_labels_batch.size(0))
    dev_acc = correct / loss_data
    logger.info('dev-loss %.3f, dev-acc %.3f' % (loss_data / total, dev_acc))


def test_one(config):
    logger_name = "my_log_predict"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(config.target_dir + "/" + config.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    fin = open(config.save_test_file, "w", encoding="utf-8")
    fstr = "ids" + "\t" + "content" + "\t" + "alternatives" + "\t" + "answer" + "\t" + "alter" + "\t" + \
           "label" + "\t" + "prediction" + "\t" + "acc" + "\n"
    fin.write(fstr)

    model = BERT.from_pretrained(config.pre_path)
    if config.is_gpu:
        model.cuda()
    tokenizer = BertTokenizer.from_pretrained(config.pre_path)
    model.load_state_dict(torch.load(config.save_dir + '/' +'siamese.pt'))

    with open(config.test_file, "r", encoding="utf-8", errors="ignore") as  f:
        for line in f.readlines():
            # sample = json.loads(line.strip())
            sample = eval(line.strip())
            ids = sample["ids"]
            content = sample["passage"]
            query = sample["query"]
            answer = sample["answer"]
            alternatives = sample["alternatives"]
            alterTemp = alternatives.split("|")
            answerTemp = answer.split("|")
            n = 0
            for alter in alterTemp:
                n += 1
                idsTemp = ids
                if alter in answerTemp:
                    label = 1
                else:
                    label = 0
                input_ids, input_mask, segment_ids = convert_features(content, alter,
                                                                      tokenizer, config.max_length_seq,
                                                                      config.max_length_query)
                if config.is_gpu:
                    input_ids_batch, input_mask_batch, segment_ids_batch = Variable(torch.LongTensor(np.array(input_ids.reshape(1, config.max_length_seq))).cuda()),  \
                                        Variable(torch.LongTensor(np.array(input_mask.reshape(1, config.max_length_seq))).cuda()), \
                                        Variable(torch.LongTensor(np.array(segment_ids.reshape(1, config.max_length_seq))).cuda())
                else:
                    input_ids_batch, input_mask_batch, segment_ids_batch = Variable(torch.LongTensor(np.array(input_ids.reshape(1, config.max_length_seq)))), \
                                        Variable(torch.LongTensor(np.array(input_mask.reshape(1,config.max_length_seq)))), \
                                        Variable(torch.LongTensor(np.array(segment_ids.reshape(1,config.max_length_seq))))

                dev_out = model(input_ids_batch, input_mask_batch, segment_ids_batch)
                _, dev_predict = dev_out.data.max(dim=1)
                prediction = np.array(dev_predict[0])
                if label == prediction:
                    ac = 1
                else:
                    ac = 0
                print(idsTemp, ":", ac)
                fstr = idsTemp + "\t" + content + "\t" + alternatives + "\t" + answer + "\t" + alter + "\t" + \
                       str(label) + "\t" + str(prediction) + "\t" + str(ac) + "\n"
                fin.write(fstr)
"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/8/31 14:46
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
from BIMPM.BIMPM_Torch.BiMPM_Prepro import data_loader, convert_features, read_dictionary, get_seg
from BIMPM.BIMPM_Torch.BiMPM_Model import BIMPM


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

    logger.info("load word_mat ...")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)

    logger.info("load char_mat ...")
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    logger.info('loading data...')
    train_loader, train_num = data_loader(config.train_eval,
                                          config.batch_size,
                                          config.max_length_content,
                                          shuffle=True)
    dev_loader, dev_num = data_loader(config.dev_eval,
                                      config.batch_size,
                                      config.max_length_content,
                                      shuffle=False)

    logger.info('train size # sent ' + str(len(train_loader)))
    logger.info('dev size # sent ' + str(len(dev_loader)))
    model = BIMPM(config, word_mat, char_mat)
    if config.is_gpu:
        model.cuda()
    para = model.parameters()
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(para, lr=config.learning_rate)
    else:
        logger.info('No Optimizer.')
        sys.exit()
    criterion = nn.CrossEntropyLoss()
    logger.info('start to train...')

    total = 0.
    correct = 0.
    loss_data = 0.
    timer = time.time()
    best_acc = 0.
    early_stop = 0
    for epoch in range(config.epoch):
        for i, (data1, data2, data1_char, data2_char, labels) in enumerate(train_loader):
            if config.is_gpu:
                data1_batch, data2_batch, data1_char_batch, data2_char_batch, labels_batch = Variable(data1.cuda()),\
                    Variable(data2.cuda()), Variable(data1_char.cuda()), Variable(data2_char.cuda()), Variable(labels.cuda())
            else:
                data1_batch, data2_batch, data1_char_batch, data2_char_batch, labels_batch = Variable(data1), \
                    Variable(data2), Variable(data1_char), Variable(data2_char), Variable(labels)
            out = model(data1_batch, data2_batch, data1_char_batch, data2_char_batch)
            optimizer.zero_grad()
            loss = criterion(out, labels_batch)
            loss.backward()
            optimizer.step()

            _, predict = out.data.max(dim=1)
            total += labels_batch.size(0)
            correct += (torch.sum(predict == labels_batch.data)).cpu().numpy()
            loss_data += (loss.data * labels_batch.size(0))

            if (i + 1) % config.display_interval == 0 or i + 2 == train_num:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, train-loss %.3f, time %.2fs, ' %
                            (epoch, i + 1, len(data1_batch), correct / total, loss_data / total,
                             time.time()-timer))
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
                model.eval()
                dev_correct = 0.
                dev_total = 0.
                dev_loss_data = 0.
                for i, (data1, data2, data1_char, data2_char, labels) in enumerate(dev_loader):
                    if config.is_gpu:
                        dev_data1_batch, dev_data2_batch, dev_data1_char_batch, dev_data2_char_batch, dev_labels_batch = Variable(data1.cuda()), \
                                Variable(data2.cuda()), Variable(data1_char.cuda()), Variable(data2_char.cuda()), Variable(labels.cuda())
                    else:
                        dev_data1_batch, dev_data2_batch, dev_data1_char_batch, dev_data2_char_batch, dev_labels_batch = Variable(data1), \
                                Variable(data2), Variable(data1_char), Variable(data2_char), Variable(labels)
                    dev_out = model(dev_data1_batch, dev_data2_batch, dev_data1_char_batch, dev_data2_char_batch)
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

    logger.info("load word_mat ...")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    logger.info("load char_mat ...")
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    logger.info('loading data...')
    dev_loader, dev_num = data_loader(config.dev_eval,
                                      config.batch_size,
                                      config.max_length_content,
                                      shuffle=False)
    logger.info('dev size # sent ' + str(len(dev_loader)))

    model = BIMPM(config, word_mat, char_mat)
    if config.is_gpu:
        model.cuda()
    model.load_state_dict(torch.load(config.save_dir + '/' +'siamese.pt'))
    model.eval()
    correct = 0.
    total = 0.
    loss_data = 0.
    criterion = nn.CrossEntropyLoss()
    for i, (data1, data2, data1_char, data2_char, labels) in enumerate(dev_loader):
        if config.is_gpu:
            dev_data1_batch, dev_data2_batch, dev_data1_char_batch, dev_data2_char_batch, dev_labels_batch = Variable(data1.cuda()), \
                    Variable(data2.cuda()), Variable(data1_char.cuda()), Variable(data2_char.cuda()), Variable(labels.cuda())
        else:
            dev_data1_batch, dev_data2_batch, dev_data1_char_batch, dev_data2_char_batch, dev_labels_batch = Variable(data1), \
                    Variable(data2), Variable(data1_char), Variable(data2_char), Variable(labels)

        dev_out = model(dev_data1_batch, dev_data2_batch, dev_data1_char_batch, dev_data2_char_batch)
        dev_loss = criterion(dev_out, dev_labels_batch)
        _, dev_predict = dev_out.data.max(dim=1)
        total += dev_labels_batch.size(0)
        correct += (torch.sum(dev_predict == dev_labels_batch.data)).cpu().numpy()
        loss_data += (dev_loss.data * dev_labels_batch.size(0))
    dev_acc = correct / total
    logger.info('dev-loss %.3f, dev-acc %.3f' % (loss_data/total, dev_acc))



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

    logger.info("load word_mat ...")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    logger.info("load word_dictionary ...")
    word2idx_dict = read_dictionary(config.word_dictionary)

    logger.info("load char_mat ...")
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    logger.info("load char_dictionary ...")
    char2idx_dict = read_dictionary(config.char_dictionary)

    model = BIMPM(config, word_mat, char_mat)
    if config.is_gpu:
        model.cuda()
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
                contentA = get_seg(content)
                contentB = get_seg(alter)
                contentA_idxs, contentB_idxs, contentA_char_idxs, \
                contentB_char_idxs = convert_features(contentA, contentB, word2idx_dict,
                                                      char2idx_dict, config.max_length_content, config.char_limit)
                if config.is_gpu:
                    data1_batch, data2_batch, data1_char_batch,\
                    data2_char_batch = Variable(torch.LongTensor(np.array(contentA_idxs.reshape(1, config.max_length_content))).cuda()),  \
                                       Variable(torch.LongTensor(np.array(contentB_idxs.reshape(1, config.max_length_content))).cuda()), \
                                       Variable(torch.LongTensor(np.array(contentA_char_idxs.reshape(1, config.max_length_content, config.char_limit))).cuda()),\
                                       Variable(torch.LongTensor(np.array(contentB_char_idxs.reshape(1, config.max_length_content, config.char_limit))).cuda())
                else:
                    data1_batch, data2_batch, data1_char_batch, \
                    data2_char_batch = Variable(torch.LongTensor(np.array(contentA_idxs.reshape(1, config.max_length_content)))), \
                                       Variable(torch.LongTensor(np.array(contentB_idxs.reshape(1, config.max_length_content)))), \
                                       Variable(torch.LongTensor(np.array(contentA_char_idxs.reshape(1, config.max_length_content, config.char_limit)))), \
                                       Variable(torch.LongTensor(np.array(contentB_char_idxs.reshape(1, config.max_length_content, config.char_limit))))

                dev_out = model(data1_batch, data2_batch, data1_char_batch, data2_char_batch)
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




"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/8/24 16:30
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
from SIAMESE.SIAMESE_Torch.SIAMESE_Prepro import data_loader, read_dictionary, get_seg, convert_features
from SIAMESE.SIAMESE_Torch.SIAMESE_Model import SIAMESE


def gradClamp(parameters, clip=5):
    for p in parameters:
        p.grad.data.clamp_(max=clip)


def train(config):
    # 定义日志格式，以及日志文件保存路径，方便以后通过日志查看模型运行到什么程度
    logger_name = "my_log"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(config.target_dir + "/" + config.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    # 加载词向量文件
    logger.info("load word_mat ...")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    # 加载训练集数据和验证集数据，并生成模型需要的batch数据
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
    # 模型初始化
    model = SIAMESE(config, word_mat)
    # 判断使用GPU训练还是CPU训练
    if config.is_gpu:
        model.cuda()
    # 创建模型优化器（本模型使用adam优化器），定义损失函数（本模型使用交叉熵损失函数）
    para = model.parameters()
    optimizer = optim.Adam(para, lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    # 开始训练模型
    logger.info('start to train...')
    total = 0.
    correct = 0.
    loss_data = 0.
    timer = time.time()
    best_acc = 0.
    early_stop = 0
    for epoch in range(config.epoch):
        if early_stop > config.early_stop:
            logger.info('training end!')
            break
        for i, (data1, data2, lengths1, lengths2, labels) in enumerate(train_loader):
            if config.is_gpu:
                data1_batch, data2_batch, labels_batch = Variable(data1.cuda()),\
                                                         Variable(data2.cuda()), \
                                                         Variable(labels.cuda())
            else:
                data1_batch, data2_batch, labels_batch = Variable(data1), Variable(data2), Variable(labels)
            out = model(data1_batch, data2_batch)
            optimizer.zero_grad()
            loss = criterion(out, labels_batch)
            loss.backward()
            # gradClamp(para, clip=config.max_grad_norm)
            optimizer.step()
            _, predict = out.data.max(dim=1)
            total += labels_batch.size(0)
            correct += (torch.sum(predict == labels_batch.data)).cpu().numpy()
            loss_data += (loss.data * labels_batch.size(0))
            # 当训练集训练了规定次数，或者一个epoch训练完成之后，进行验证集验证
            if (i + 1) % config.display_interval == 0 or i + 2 == train_num:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, train-loss %.3f, time %.2fs, ' %
                            (epoch, i + 1, len(data1_batch), correct / total, loss_data / total,
                             time.time()-timer))
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
                # 验证时，为保证模型dropout被关掉，使用model.eval()
                model.eval()
                dev_correct = 0.
                dev_total = 0.
                dev_loss_data = 0.
                for i, (data1, data2, lengths1, lengths2, labels) in enumerate(dev_loader):
                    if config.is_gpu:
                        dev_data1_batch, dev_data2_batch, dev_labels_batch = Variable(data1.cuda()), \
                                                                             Variable(data2.cuda()), \
                                                                             Variable(labels.cuda())
                    else:
                        dev_data1_batch, dev_data2_batch, dev_labels_batch = Variable(data1), Variable(data2), Variable(labels)
                    dev_out = model(dev_data1_batch, dev_data2_batch)
                    dev_loss = criterion(dev_out, dev_labels_batch)
                    _, dev_predict = dev_out.data.max(dim=1)
                    dev_total += dev_labels_batch.size(0)
                    dev_correct += (torch.sum(dev_predict == dev_labels_batch.data)).cpu().numpy()
                    dev_loss_data += (dev_loss.data * dev_labels_batch.size(0))

                dev_acc = dev_correct / dev_total
                logger.info('dev-loss %.3f, dev-acc %.3f' % (dev_loss_data / dev_total, dev_acc))
                # 为防止模型过拟合，采用及早停止方式，当验证集在规定次数内没有提升时，停止模型训练
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
                # 注意：每次验证完之后，要把dropout打开，使用model.train()
                model.train()
    logger.info('training end!')


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
    # 测试集结果保存文件
    fin = open(config.save_test_file, "w", encoding="utf-8")
    fstr = "ids" + "\t" + "content" + "\t" + "alternatives" + "\t" + "answer" + "\t" + "alter" + "\t" + \
           "label" + "\t" + "prediction" + "\t" + "acc" + "\n"
    fin.write(fstr)
    # 加载词向量及词典
    logger.info("load word_mat ...")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    logger.info("load word_dictionary ...")
    word2idx_dict = read_dictionary(config.word_dictionary)
    # 模型初始化
    model = SIAMESE(config, word_mat)
    if config.is_gpu:
        model.cuda()
    # 加载之前训练好的模型
    model.load_state_dict(torch.load(config.save_dir + '/' +'siamese.pt'))
    # 开始预测
    with open(config.test_file, "r", encoding="utf-8", errors="ignore") as  f:
        # 对测试集中每一条数据进行单独预测
        for line in f.readlines():
            sample = eval(line.strip())
            ids = sample["ids"]
            content = sample["passage"]
            query = sample["query"]
            answer = sample["answer"]
            alternatives = sample["alternatives"]
            alterTemp = alternatives.split("|")
            answerTemp = answer.split("|")
            # 将每个选项与用户回答，进行文本蕴含预测，判断该选项是否正确
            for alter in alterTemp:
                idsTemp = ids
                if alter in answerTemp:
                    label = 1
                else:
                    label = 0
                contentA = get_seg(content)
                contentB = get_seg(alter)
                # 数据转换，将字符型文本，转换成token_id形式
                contentA_idxs, contentB_idxs, _, _ = convert_features(contentA, contentB, word2idx_dict, config.max_length_content)
                if config.is_gpu:
                    data1_batch, data2_batch = Variable(torch.LongTensor(np.array(contentA_idxs.reshape(1, config.max_length_content))).cuda()),  \
                                       Variable(torch.LongTensor(np.array(contentB_idxs.reshape(1, config.max_length_content))).cuda())
                else:
                    data1_batch, data2_batch = Variable(torch.LongTensor(np.array(contentA_idxs.reshape(1, config.max_length_content)))), \
                                       Variable(torch.LongTensor(np.array(contentB_idxs.reshape(1, config.max_length_content))))

                # 模型预测
                test_out = model(data1_batch, data2_batch)
                _, test_predict = test_out.data.max(dim=1)
                prediction = np.array(test_predict[0])
                if label == prediction:
                    ac = 1.0
                else:
                    ac = 0.0
                print(idsTemp, ":", ac)
                fstr = idsTemp + "\t" + content + "\t" + alternatives + "\t" + answer + "\t" + alter + "\t" + \
                       str(label) + "\t" + str(prediction) + "\t" + str(ac) + "\n"
                fin.write(fstr)


def predict(config):
    # 加载词向量及词典
    print("load word_mat ...")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    print("load word_dictionary ...")
    word2idx_dict = read_dictionary(config.word_dictionary)
    # 模型初始化
    model = SIAMESE(config, word_mat)
    if config.is_gpu:
        model.cuda()
    # 加载之前训练好的模型
    model.load_state_dict(torch.load(config.save_dir + '/' +'siamese.pt'))
    # 开始预测
    with open(config.test_file, "r", encoding="utf-8", errors="ignore") as  f:
        # 对测试集中每一条数据进行单独预测
        n_whole_total = 0
        n_total = 0
        acc_total = 0
        acc_whole_total = 0
        for line in f.readlines():
            n_whole_total +=1
            sample = eval(line.strip())
            content = sample["passage"]
            answer = sample["answer"]
            alternatives = sample["alternatives"]
            alterTemp = alternatives.split("|")
            answerTemp = answer.split("|")
            ac_whole = []
            # 将每个选项与用户回答，进行文本蕴含预测，判断该选项是否正确
            for alter in alterTemp:
                n_total += 1
                if alter in answerTemp:
                    label = 1
                else:
                    label = 0
                contentA = get_seg(content)
                contentB = get_seg(alter)
                # 数据转换，将字符型文本，转换成token_id形式
                contentA_idxs, contentB_idxs, _, _ = convert_features(contentA, contentB, word2idx_dict, config.max_length_content)
                if config.is_gpu:
                    data1_batch, data2_batch = Variable(torch.LongTensor(np.array(contentA_idxs.reshape(1, config.max_length_content))).cuda()),  \
                                       Variable(torch.LongTensor(np.array(contentB_idxs.reshape(1, config.max_length_content))).cuda())
                else:
                    data1_batch, data2_batch = Variable(torch.LongTensor(np.array(contentA_idxs.reshape(1, config.max_length_content)))), \
                                       Variable(torch.LongTensor(np.array(contentB_idxs.reshape(1, config.max_length_content))))

                # 模型预测
                test_out = model(data1_batch, data2_batch)
                _, test_predict = test_out.data.max(dim=1)
                prediction = np.array(test_predict[0])
                if label == prediction:
                    ac = 1.0
                else:
                    ac = 0.0
                acc_total += ac
                ac_whole.append(ac)
            acc_whole = np.mean(ac_whole)
            if acc_whole == 1.0:
                acc_whole_total += 1.0
        acc_whole_total = acc_whole_total/n_whole_total
        acc_total = acc_total / n_total
        print("准确率：", acc_total)
        print("单条样本准确率：", acc_whole_total)

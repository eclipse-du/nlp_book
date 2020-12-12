import torch
import torch.nn as nn
import time
import numpy as np
import sys
from SkipGramModel import SkipGramModel
from dataset import Word2vecdataset




def train(config):
    # 定义日志格式，以及日志文件保存路径，方便以后通过日志查看模型运行到什么程度
    d = Word2vecdataset('news_seg.txt',K=20, window_size=3, min_count=100)
    vocab_size = d.vocab_size
    embed_size = 100
    batch_size = 4096
    batch_iter = d.build_batches(batch_size)
    use_cuda = True
    lr = 0.1
    lr_decay = 0.8
    sg = SkipGramModel(d.vocab_size, embed_size, use_cuda)
    optimizer = torch.optim.Adagrad(sg.parameters(), lr=lr)
    for i, (c, pos_index_list, neg_index_list) in enumerate(batch_iter):
        pos_c = torch.Tensor(c).long()
        pos_v = torch.Tensor(pos_index_list).long()
        neg_v = torch.Tensor(neg_index_list).long()
        if use_cuda:
            pos_c, pos_v, neg_v = pos_c.cuda(), pos_v.cuda(), neg_v.cuda()
        optimizer.zero_grad()
        loss = sg(pos_c, pos_v, neg_v)
        loss.backward()
        optimizer.step()
    if i % 1000 == 999:
        print('loss:', loss.data.tolist())
        lr *= lr_decay
        adjust_learning_rate(optimizer, lr)
    if i % 10000 == 9999:
        sg.save_embedding('cn_news_word2vec.txt', d.word2index)


def adjust_learning_rate(optimizer, lr):
    """学习率手动调整.
    Args:
        optimizer[torch.optim.adagrad] : 模型使用梯度集成类
        lr[float]                      : 修改的学习率
    Returns:
        None
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cal_top_k_sim(self, word, K=5):
    """计算指定单词最相似的的topK单词
    Args:
        word[str]                    : 指定单词文本
        K[int]                       : topK取值
    Returns:
        result[list((str,float))]    : 与指定单词最相似的topK单词集其相似度
    """
    if word not in self.word2index:
        print('word does not exist in the embedding file')
        return
    if K < 1:
        print('K must bigger or equal to 1!')
    idx = self.word2index[word]
    word_array = self.embedding_array[idx, :]
    result = []
    scores = np.dot(self.embedding_array, word_array) / \
        (self.vector_normal[idx]*self.vector_normal)
    word_indexs = np.argsort(scores)
    print('word:', word)
    result = []
    for word_index in word_indexs[::-1][1:K+1]:
        pair = (self.word2index.inverse[word_index], scores[word_index])
        print(pair)
        result.append(pair)
    return result

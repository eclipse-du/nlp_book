import torch
import torch.nn as nn
import time
import numpy as np
import sys
from gloveModel import GloveModel
from dataset import Glovedataset




def main():
    ''' main function for glove trainning
    '''
    base_dir = 'd:/data/word_embedding/'
    g = Glovedataset(base_dir+'news_seg.txt', window_size=7,
                     min_count=100, x_max=1000)
    embed_size = 100
    use_gpu = True
    batch_size = 4096
    iter_size = 20000
    print('start train')
    gm = GloveModel(g.vocab_size, embed_size, use_gpu)
    l_r = 0.5
    l_r_dacay = 0.8
    optimizer = torch.optim.Adagrad(gm.parameters(), lr=l_r)
    time_start = time.time()
    for i in range(iter_size):
        optimizer.zero_grad()
        pos_i, pos_j, wij, wf = g.next_batch(batch_size)
        if use_gpu:
            pos_i = torch.tensor(pos_i).long().cuda()
            pos_j = torch.tensor(pos_j).long().cuda()
            wij = torch.tensor(wij).float().cuda()
            wf = torch.tensor(wf).float().cuda()
        else:
            pos_i = torch.tensor(pos_i).long()
            pos_j = torch.tensor(pos_j).long()
            wij = torch.tensor(wij).float()
            wf = torch.tensor(wf).float()
        loss = gm(pos_i, pos_j, wij, wf)
        if i % 100 == 0:
            print(loss)
            l_r *= l_r_dacay
            adjust_learning_rate(optimizer, l_r)
        if i % 1000 == 999:
            time_end = time.time()
            print('finish! Time cost:', time_end-time_start)
            gm.save_embedding(base_dir+'glove.txt', g.word2index)
            time_start = time.time()
        loss.backward()
        optimizer.step()



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

import jieba
from bidict import bidict
from collections import defaultdict
import numpy as np
import random

class Word2vecdataset():
    def __init__(self, fname='news_seg.txt',K=20, window_size=3, min_count=100):
        """
        Word2vecdataset初始化
        """
        self.fname = fname
        self.K = K
        self.window_size  = window_size
        self.min_count = min_count
        self.build_vocab()
        self.build_negative_sample_space()
        
    def build_vocab(self):
        """构建词-索引双向字典及词频统计词典
        """
        # 词-索引双向字典
        self.word2index = bidict()
        # 单词词频统计字典
        self.word_count = defaultdict(int)
        self.word2index['UNK'] = 0
        index = 1
        with open(self.fname, encoding='utf8') as f:
            for line in f:
                for word in line.split():
                    self.word_count[word] += 1
        # 过滤低频次
        remove_word_list = []
        for word, count in self.word_count.items():
            if count >= self.min_count:
                self.word2index[word] = index
                index += 1
            else:
                remove_word_list.append(word)
        for word in remove_word_list:
            self.word_count.pop(word)
        # 单词数量
        self.vocab_size = len(self.word2index.keys())
        # 为了防止负样本数量大于单词数量，导致死循环的情况出现
        # 一般训练不会出现这种情况
        if self.K >= self.vocab_size:
            self.K = max(1, self.vocab_size - self.window_size)
            
    def build_negative_sample_space(self):
        """构建负样本采样空间
        """
        M = 1e8
        self.negative_sample_space = []
        word_value = np.array(list(self.word_count.values()))**0.75
        word_value /= word_value.sum()
        word_size_in_space = np.round(word_value * M)
        for i, word in enumerate(self.word_count):
            self.negative_sample_space += [self.word2index[word]] * \
                int(word_size_in_space[i])

    def pick_negative_smaples(self, pos_index):
        """负样本采样
        Args:
            pos_index[int]    : 正样本索引值
        Returns:
            neg_samples[List] : 负样本采样列表
        """
        neg_samples = []
        while len(neg_samples) < self.K:
            neg_index = random.choice(self.negative_sample_space)
            # 避免负采样的样本隶属于正样本列表中
            if neg_index != pos_index:
                neg_samples.append(neg_index)
        return neg_samples
        
class Glovedataset():
    def __init__(self, fname='news_seg.txt',K=20, window_size=3, min_count=100,x_max=1000):
        """
        Glovedataset初始化
        """
        self.fname = fname
        self.K = K
        self.window_size  = window_size
        self.x_max = x_max
        self.min_count = min_count
        self.build_vocab()
        self.build_negative_sample_space()
        self.build_cm()
        
    def build_vocab(self):
        """构建词-索引双向字典及词频统计词典
        """
        # 词-索引双向字典
        self.word2index = bidict()
        # 单词词频统计字典
        self.word_count = defaultdict(int)
        self.word2index['UNK'] = 0
        index = 1
        with open(self.fname, encoding='utf8') as f:
            for line in f:
                for word in line.split():
                    self.word_count[word] += 1
        # 过滤低频次
        remove_word_list = []
        for word, count in self.word_count.items():
            if count >= self.min_count:
                self.word2index[word] = index
                index += 1
            else:
                remove_word_list.append(word)
        for word in remove_word_list:
            self.word_count.pop(word)
        # 单词数量
        self.vocab_size = len(self.word2index.keys())
        # 为了防止负样本数量大于单词数量，导致死循环的情况出现
        # 一般训练不会出现这种情况
        if self.K >= self.vocab_size:
            self.K = max(1, self.vocab_size - self.window_size)
            
    def build_negative_sample_space(self):
        """构建负样本采样空间
        """
        M = 1e8
        self.negative_sample_space = []
        word_value = np.array(list(self.word_count.values()))**0.75
        word_value /= word_value.sum()
        word_size_in_space = np.round(word_value * M)
        for i, word in enumerate(self.word_count):
            self.negative_sample_space += [self.word2index[word]] * \
                int(word_size_in_space[i])
    
    def build_cm(self):
        """build co-occurence matrix for the batch_data
        """
        data = [],x = [],y = []
        cm = coo_matrix(([0], ([0], [0])), shape=(self.vocab_size, self.vocab_size))
        with open(self.fname, encoding='utf8') as f:
            for line_index, line in enumerate(f):
                if line_index % 10000 == 9999:
                    print('line index', line_index)
                    index = (data, (x, y))
                    cm = cm + coo_matrix(index, shape=(
                            self.vocab_size, self.vocab_size))
                    data = [],x = [],y = []
                words = line.strip().split()
                for i, word in enumerate(words):
                    # 若中心词不在词向量中就没必要参与训练
                    if word not in self.word2index:
                        continue
                    word_index = self.word2index[word]
                    for j in range(1, min(self.window_size+1, len(words)-i)):
                        window_word = words[i+j]
                        # 若窗口词汇是unk则其index为0
                        if window_word in self.word2index:
                            window_index = self.word2index[window_word]
                        else:
                            continue
                        if word_index < window_index:
                            x.append(word_index)
                            y.append(window_index)
                        else:
                            x.append(window_index)
                            y.append(word_index)
                        data.append(1.0)
        index = (data, (x, y))
        cm = cm + coo_matrix(index, shape=(
            self.vocab_size, self.vocab_size))
        self.cm = cm.tocsr()
        save_npz(self.base_dir+'cm.npz', self.cm)

    def next_batch(self, batch_size):
        """构建下一批训练批处理样本
        Args:
            batch_size[int]   : 批处理数量
        Returns:
            batch_data[list]  : 单批次数据,返回单条数据包括
                                词i索引,词j索引,词ij共现分值,词ij共现加权分值
        """
        i = np.random.choice(self.vocab_index, size=batch_size)
        j = np.random.choice(self.vocab_index, size=batch_size)
        word_i = np.minimum(i, j)
        word_j = np.maximum(i, j)
        wij = [self.cm[word_i[x], word_j[x]].tolist() + 1
                for x in range(batch_size)]
        '''
        idx = np.random.choice(self.pair.shape[0], size=batch_size)
        # 切分索引变量
        word_i, word_j = np.hsplit(self.pair[idx], 2)
        wij = [self.cm[word_i[x][0], word_j[x][0]].tolist()
                for x in range(batch_size)]
        '''
        wf = [self.weight_func(wij[idx]) for idx in range(batch_size)]
        return word_i, word_j, wij, wf


    def pick_negative_smaples(self, pos_index):
        """负样本采样
        Args:
            pos_index[int]    : 正样本索引值
        Returns:
            neg_samples[List] : 负样本采样列表
        """
        neg_samples = []
        while len(neg_samples) < self.K:
            neg_index = random.choice(self.negative_sample_space)
            # 避免负采样的样本隶属于正样本列表中
            if neg_index != pos_index:
                neg_samples.append(neg_index)
        return neg_samples

def process_file(file_input_path, file_out_path, encoding='utf8'):
    """初始化函数
    Args:
        file_input_path[str] : 原始数据文件
        file_out_path[str]   : 输出文件
        encoding[str]   : 编码(默认utf8)
    """
    words = ''
    with open(file_input_path, encoding=encoding) as f:
        for line in f:
            # 过滤字符小于3的单行
            if len(line) < 3:
                continue
            s = ''
            for word in jieba.cut(line):
                # 过滤单字
                if len(word) > 1:
                    s += word + ' '
            words += s.strip()+'\n'
    f = open(file_out_path, mode='a', encoding='utf8')
    f.write(words)
    f.close()






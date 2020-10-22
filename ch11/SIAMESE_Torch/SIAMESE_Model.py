"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/8/24 9:57
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIAMESE(nn.Module):
    def __init__(self, config, word_mat):
        """
        模型初始化函数
        :param config: 配置项类
        :param word_mat: 外部词向量
        """
        super(SIAMESE, self).__init__()
        self.vocab_size = config.vocab_size    # 词典大小
        self.word_dims = config.word_dims   # 词向量维度
        self.hidden_dims = config.hidden_dims   # lstm隐藏节点大小
        self.num_layers = config.num_layers     # lstm层数
        self.dropout_rate = config.dropout_rate     # dropout 值
        self.word_mat = torch.from_numpy(word_mat)
        # embedding 词编码层
        self.embedding_table = nn.Embedding(num_embeddings=self.vocab_size+2, embedding_dim=self.word_dims)
        self.embedding_table.weight.data.copy_(self.word_mat)
        self.embedding_table.weight.requires_grad = True
        # 双向lstm网络
        self.bilstm = nn.LSTM(input_size=self.word_dims,
                              hidden_size=self.hidden_dims,
                              num_layers=self.num_layers,
                              batch_first=True,
                              bidirectional=True)
        # 全连接网络
        self.linear = nn.Linear(self.hidden_dims * 2 * 4, self.hidden_dims)
        self.predictor = nn.Linear(self.hidden_dims, 2)
        # softmax层
        self.softmax = nn.Softmax()

    def dropout(self, v):
        return F.dropout(v, p=self.dropout_rate, training=self.training)

    def forward(self, sentence_one_word, sentence_two_word):
        """
        模型向前传播
        :param sentence_one_word: 用户回答文本dict_id
        :param sentence_two_word: 选项文本dict_id
        :return: 预测概率
        """
        word_embedded_sentence_one = self.embedding_table(sentence_one_word)
        word_embedded_sentence_two = self.embedding_table(sentence_two_word)
        one_outputs, _ = self.bilstm(word_embedded_sentence_one)
        two_outputs, _ = self.bilstm(word_embedded_sentence_two)
        aggregated_one_f, aggregated_one_b = torch.split(one_outputs, self.hidden_dims, dim=-1)
        aggregated_two_f, aggregated_two_b = torch.split(two_outputs, self.hidden_dims, dim=-1)

        last_one_output = torch.cat([aggregated_one_f[:, -1, :],
                                aggregated_one_b[:, 0, :]], dim=-1)
        last_two_output = torch.cat([aggregated_two_f[:, -1, :],
                                aggregated_two_b[:, 0, :]], dim=-1)
        last_output = torch.cat((last_one_output, last_two_output, last_one_output*last_two_output,
                                 last_one_output-last_two_output), 1)
        feature = self.dropout(self.predictor(self.linear(last_output)))
        preds = self.softmax(feature)
        return preds




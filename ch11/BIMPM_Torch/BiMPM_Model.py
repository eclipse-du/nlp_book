"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/8/31 14:46
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BIMPM(nn.Module):
    def __init__(self, config, word_mat, char_mat):
        """
        模型初始化函数
        :param config: 配置项类
        :param word_mat: 外部词向量
        :param char_mat: 外部字向量
        """
        super(BIMPM, self).__init__()
        self.vocab_size = config.vocab_size     # 词典大小
        self.vocab_char_size = config.vocab_char_size   # 字典大小
        self.word_dims = config.word_dims   # 词向量维度
        self.char_dims = config.char_dims   # 字向量维度
        self.hidden_dims = config.hidden_dims   # bilstm隐藏节点大小
        self.char_hidden_size = config.char_hidden_size    # 字组合向量维度
        self.num_layers = config.num_layers     # bilstm层数
        self.dropout_rate = config.dropout_rate     # dropout 值
        self.epsilon = config.epsilon
        self.char_limit = config.char_limit     # 词包含字数限制
        # Word or char Representation Layer 字词表示层
        self.embedding_word = nn.Embedding(num_embeddings=self.vocab_size+2, embedding_dim=self.word_dims)
        self.embedding_char = nn.Embedding(num_embeddings=self.vocab_char_size+2, embedding_dim=self.char_dims)
        # initialize word or char embedding 初始化字词向量
        self.word_mat = torch.from_numpy(word_mat)
        self.char_mat = torch.from_numpy(char_mat)
        self.embedding_word.weight.data.copy_(self.word_mat)
        self.embedding_word.weight.requires_grad = True
        self.embedding_char.weight.data.copy_(self.char_mat)
        self.embedding_char.weight.requires_grad = True
        # Char Representation Layer 字组合向量表示层
        self.char_LSTM = nn.LSTM(input_size=self.char_dims,
                                 hidden_size=self.char_hidden_size,
                                 num_layers=self.num_layers,
                                 bidirectional=False,
                                 batch_first=True)
        # Context Representation Layer 上下文表示层
        self.context_hidden_size = self.char_hidden_size + self.word_dims
        self.context_LSTM = nn.LSTM(input_size=self.context_hidden_size,
                                    hidden_size=self.hidden_dims,
                                    num_layers=self.num_layers,
                                    bidirectional=True,
                                    batch_first=True)
        # Multi-perspective Matching Layer 匹配层
        self.num_perspective = config.num_perspective   # 视角个数
        for i in range(1, 9):
            setattr(self, 'mpm_w%d'%i, nn.Parameter(torch.rand(self.num_perspective, self.hidden_dims)))
        # Aggregation Layer 融合层
        self.aggregation_LSTM = nn.LSTM(input_size=self.num_perspective * 8,
                                        hidden_size=self.hidden_dims,
                                        num_layers=self.num_layers,
                                        bidirectional=True,
                                        batch_first=True)
        # Prediction Layer 预测层
        self.pred_fc1 = nn.Linear(self.hidden_dims * 4, self.hidden_dims * 2)
        self.pred_fc2 = nn.Linear(self.hidden_dims * 2, 2)
        self.reset_parameters()

    def reset_parameters(self):
        # Char Representation Layer
        nn.init.xavier_uniform(self.char_LSTM.weight_ih_l0)
        nn.init.constant(self.char_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.char_LSTM.weight_hh_l0)
        nn.init.constant(self.char_LSTM.bias_hh_l0, val=0)
        # Context Representation Layer
        nn.init.xavier_uniform(self.context_LSTM.weight_ih_l0)
        nn.init.constant(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0)
        nn.init.constant(self.context_LSTM.bias_hh_l0, val=0)
        nn.init.xavier_uniform(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant(self.context_LSTM.bias_hh_l0_reverse, val=0)
        # Multi-perspective Matching Layer
        for i in range(1, 9):
            nn.init.kaiming_normal(getattr(self, 'mpm_w%d'%i))
        # ----- Aggregation Layer -----
        nn.init.xavier_uniform(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant(self.aggregation_LSTM.bias_hh_l0, val=0)
        nn.init.xavier_uniform(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)
        # ----- Prediction Layer ----
        nn.init.uniform(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc1.bias, val=0)
        nn.init.uniform(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant(self.pred_fc2.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=self.dropout_rate, training=self.training)

    def multi_perspective_matching(self, p_fw, p_bw, q_fw, q_bw):
        """
        对两句话进行多策略多视角匹配
        :param p_fw: 句子p经过bilstm的前向句子矩阵
        :param p_bw: 句子p经过bilstm的反向句子矩阵
        :param q_fw: 句子q经过bilstm的前向句子矩阵
        :param q_bw: 句子q经过bilstm的反向句子矩阵
        :return: 匹配结果
        """
        def _cosine_similarity(x1, x2, epsilon=1e-6):
            # 计算两个向量的cosine相似度
            cos = (x1 * x2)
            cos = cos.sum(-1)
            x1_norm = torch.sqrt(torch.sum(x1 ** 2, -1).clamp(min=epsilon))
            x2_norm = torch.sqrt(torch.sum(x2 ** 2, -1).clamp(min=epsilon))
            cos = cos / x1_norm / x2_norm
            return cos

        def _cosine_matrix(x1, x2, epsilon=1e-6):
            # 计算两个向量的cosine相似度矩阵
            x1 = x1.unsqueeze(2)
            x2 = x2.unsqueeze(1)
            cos_matrix = _cosine_similarity(x1, x2, epsilon)
            return cos_matrix

        def _mean_attentive_vectors(x2, cosine_matrix, epsilon=1e-6):
            # 平均注意力相似匹配矩阵
            expanded_cosine_matrix = cosine_matrix.unsqueeze(-1)
            x2 = x2.unsqueeze(1)
            weighted_sum = (expanded_cosine_matrix * x2).sum(2)
            sum_cosine = (cosine_matrix.sum(-1) + epsilon).unsqueeze(-1)
            attentive_vector = weighted_sum / sum_cosine
            return attentive_vector

        def _max_attentive_vectors(x2, cosine_matrix):
            # 最大注意力相似匹配矩阵
            _, max_x2_step = cosine_matrix.max(-1)
            embedding_size = x2.size()[-1]
            timesteps = max_x2_step.size()[-1]
            x2 = x2.contiguous().view(-1, embedding_size)
            max_x2_step = max_x2_step.contiguous().view(-1)
            max_x2 = x2[max_x2_step]
            attentive_vector = max_x2.view(-1, timesteps, embedding_size)
            return attentive_vector

        def _time_distributed_multiply(x, w, perspective):
            # 多视角匹配
            n_dim = x.dim()
            embedding_size = x.size()[-1]
            timesteps = x.size()[1]
            x = x.contiguous().view(-1, embedding_size)
            x = torch.unsqueeze(x, 1)
            w = torch.unsqueeze(w, 0)
            x = x * w
            if n_dim == 3:
                x = x.view(-1, timesteps, perspective, embedding_size)
            elif n_dim == 2:
                x = x.view(-1, perspective, embedding_size)
            return x

        def _full_matching(h1, h2, w, perspective=20, epsilon=1e-6):
            # 全匹配
            h2_last_state = h2[:, -1, :]
            h1 = _time_distributed_multiply(h1, w, perspective)
            h2 = _time_distributed_multiply(h2_last_state, w, perspective)
            h2 = h2.unsqueeze(1)
            matching = _cosine_similarity(h1, h2, epsilon)
            return matching

        def _max_pooling_matching(h1, h2, w, perspective=20, epsilon=1e-6):
            # 最大池化匹配
            h1 = _time_distributed_multiply(h1, w, perspective)
            h2 = _time_distributed_multiply(h2, w, perspective)
            h1 = h1.unsqueeze(2)
            h2 = h2.unsqueeze(1)
            cos = _cosine_similarity(h1, h2, epsilon)
            matching = cos.max(2)[0]
            return matching

        def _attentive_matching(h1, h2, cosine_matrix, w, perspective=20, epsilon=1e-6):
            # 注意力匹配
            h1 = _time_distributed_multiply(h1, w, perspective)
            attentive_vec = _mean_attentive_vectors(h2, cosine_matrix, epsilon)
            attentive_vec = _time_distributed_multiply(attentive_vec, w, perspective)
            matching = _cosine_similarity(h1, attentive_vec, epsilon)
            return matching

        def _max_attentive_matching(h1, h2, cosine_matrix, w, perspective=20, epsilon=1e-6):
            # 最大注意力匹配
            h1 = _time_distributed_multiply(h1, w, perspective)
            max_attentive_vec = _max_attentive_vectors(h2, cosine_matrix)
            max_attentive_vec = _time_distributed_multiply(max_attentive_vec, w, perspective)
            matching = _cosine_similarity(h1, max_attentive_vec, epsilon)
            return matching

        list_of_each_perspective = []
        matching_result_fw = _full_matching(p_fw, q_fw, self.mpm_w1, self.num_perspective, self.epsilon)
        matching_result_bw = _full_matching(p_bw, q_bw, self.mpm_w2, self.num_perspective, self.epsilon)
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching1 = torch.cat(list_of_each_perspective, -1)

        cosine_matrix_fw = _cosine_matrix(p_fw, q_fw)
        cosine_matrix_bw = _cosine_matrix(p_bw, q_bw)
        matching_result_fw = _attentive_matching(p_fw, q_fw, cosine_matrix_fw, self.mpm_w3, self.num_perspective, self.epsilon)
        matching_result_bw = _attentive_matching(p_bw, q_bw, cosine_matrix_bw, self.mpm_w4, self.num_perspective, self.epsilon)
        list_of_each_perspective = []
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching2 = torch.cat(list_of_each_perspective, -1)

        matching_result_fw = _max_attentive_matching(p_fw, q_fw, cosine_matrix_fw, self.mpm_w5, self.num_perspective, self.epsilon)
        matching_result_bw = _max_attentive_matching(p_bw, q_bw, cosine_matrix_bw, self.mpm_w6, self.num_perspective, self.epsilon)
        list_of_each_perspective = []
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching3 = torch.cat(list_of_each_perspective, -1)

        matching_result_fw = _max_pooling_matching(p_fw, q_fw, self.mpm_w7, self.num_perspective, self.epsilon)
        matching_result_bw = _max_pooling_matching(p_bw, q_bw, self.mpm_w8, self.num_perspective, self.epsilon)
        list_of_each_perspective = []
        list_of_each_perspective.append(matching_result_fw)
        list_of_each_perspective.append(matching_result_bw)
        maching4 = torch.cat(list_of_each_perspective, -1)

        return torch.cat([maching1, maching2, maching3, maching4], dim=-1)

    def forward(self, contentA_idxs, contentB_idxs, contentA_char_idxs, contentB_char_idxs):
        """
        模型向前传播
        :param contentA_idxs: 用户回答文本词token_id
        :param contentB_idxs: 选项文本文本词token_id
        :param contentA_char_idxs: 用户回答文本字token_id
        :param contentB_char_idxs: 选项文本文本字token_id
        :return: 预测概率
        """
        # Word Representation Layer
        word_embedded_sentence_one = self.embedding_word(contentA_idxs)
        word_embedded_sentence_two = self.embedding_word(contentB_idxs)
        seq_len_p = contentA_char_idxs.size(1)
        seq_len_h = contentB_char_idxs.size(1)
        char_one = contentA_char_idxs.view(-1, self.char_limit)
        char_two = contentB_char_idxs.view(-1, self.char_limit)
        char_embedded_sentence_one = self.embedding_char(char_one)
        char_embedded_sentence_two = self.embedding_char(char_two)

        # Char Representation Layer
        _, (char_lstm_sentence_one, _) = self.char_LSTM(char_embedded_sentence_one)
        _, (char_lstm_sentence_two, _) = self.char_LSTM(char_embedded_sentence_two)
        char_lstm_sentence_one = char_lstm_sentence_one.view(-1, seq_len_p, self.char_hidden_size)
        char_lstm_sentence_two = char_lstm_sentence_two.view(-1, seq_len_h, self.char_hidden_size)

        # Context Representation Layer
        embedded_sentence_one = torch.cat([word_embedded_sentence_one, char_lstm_sentence_one], dim=-1)
        embedded_sentence_two = torch.cat([word_embedded_sentence_two, char_lstm_sentence_two], dim=-1)
        embedded_sentence_one = self.dropout(embedded_sentence_one)
        embedded_sentence_two = self.dropout(embedded_sentence_two)
        context_lstm_sentence_one, _ = self.context_LSTM(embedded_sentence_one)
        context_lstm_sentence_two, _ = self.context_LSTM(embedded_sentence_two)
        context_lstm_sentence_one = self.dropout(context_lstm_sentence_one)
        context_lstm_sentence_two = self.dropout(context_lstm_sentence_two)

        # Multi-perspective Matching Layer
        context_lstm_sentence_one_f, context_lstm_sentence_one_b = torch.split(context_lstm_sentence_one,
                                                                               self.hidden_dims, dim=-1)
        context_lstm_sentence_two_f, context_lstm_sentence_two_b = torch.split(context_lstm_sentence_two,
                                                                               self.hidden_dims, dim=-1)
        maching_one = self.multi_perspective_matching(context_lstm_sentence_one_f,
                                                      context_lstm_sentence_one_b,
                                                      context_lstm_sentence_two_f,
                                                      context_lstm_sentence_two_b)
        maching_two = self.multi_perspective_matching(context_lstm_sentence_two_f,
                                                      context_lstm_sentence_two_b,
                                                      context_lstm_sentence_one_f,
                                                      context_lstm_sentence_one_b)
        maching_one = self.dropout(maching_one)
        maching_two = self.dropout(maching_two)

        # Aggregation Layer
        aggregated_one, _ = self.aggregation_LSTM(maching_one)
        aggregated_two, _ = self.aggregation_LSTM(maching_two)
        aggregated_one_f, aggregated_one_b = torch.split(aggregated_one, self.hidden_dims, dim=-1)
        aggregated_two_f, aggregated_two_b = torch.split(aggregated_two, self.hidden_dims, dim=-1)
        aggregated = torch.cat([aggregated_one_f[:, -1, :],
                                aggregated_one_b[:, 0, :],
                                aggregated_two_f[:, -1, :],
                                aggregated_two_b[:, 0, :]], dim=-1)
        aggregated = self.dropout(aggregated)

        # Prediction Layer
        x = F.tanh(self.pred_fc1(aggregated))
        x = self.dropout(x)
        x = self.pred_fc2(x)
        return x



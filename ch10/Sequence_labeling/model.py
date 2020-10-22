import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


def to_scalar(var):  # var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):  # vec是1*5, type是Variable

    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # vec.size()维度是1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))  # 为什么指数之后再求和，而后才log呢


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, start_tag, end_tag):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.start_tag = start_tag
        self.end_tag = end_tag

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j. 居然是随机初始化的！！！！！！！！！！！！！！！之后的使用也是用这随机初始化的值进行操作！！
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[self.start_tag ], :] = -10000
        self.transitions.data[:, tag_to_ix[self.end_tag]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    # 预测序列的得分
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)  # 1*5 而且全是-10000

        # self.start_tag  has all of the score.
        init_alphas[0][self.tag_to_ix[
            self.start_tag ]] = 0.  # 因为start tag是4，所以tensor([[-10000., -10000., -10000.,      0., -10000.]])，将start的值为零，表示开始进行网络的传播，

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)  # 初始状态的forward_var，随着step t变化

        # Iterate through the sentence 会迭代feats的行数次，
        for feat in feats:  # feat的维度是５ 依次把每一行取出来~
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):  # next tag 就是简单 i，从0到len
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1,
                                                               self.tagset_size)  # 维度是1*5 噢噢！原来，LSTM后的那个矩阵，就被当做是emit score了

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)  # 维度是１＊５
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # 第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).unsqueeze(0))
            # 此时的alphas t 是一个长度为5，例如<class 'list'>: [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)  # 到第（t-1）step时５个标签的各自分数
        terminal_var = forward_var + self.transitions[self.tag_to_ix[
            self.end_tag]]  # 最后只将最后一个单词的forward var与转移 stop tag的概率相加 tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
        alpha = log_sum_exp(terminal_var)  # alpha是一个0维的tensor

        return alpha

    # 得到feats
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.word_embeds(sentence)

        embeds = embeds.unsqueeze(1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # 11*1*4
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # 11*4

        lstm_feats = self.hidden2tag(lstm_out)  # 11*5 is a linear layer

        return lstm_feats

    # 得到gold_seq tag的score 即根据真实的label 来计算一个score，但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence #feats 11*5  tag 11 维
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[self.start_tag ]]), tags])  # 将self.start_tag 的标签３拼接到tag序列最前面，这样tag就是12个了

        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, self.start_tag , END_TAG, 取对应标签的值
            # transition【j,i】 就是从i ->j 的转移概率值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.end_tag], tags[-1]]
        return score

    # 解码，得到预测的序列，以及预测序列的得分
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[self.start_tag ]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]  # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # 从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t)  # bptrs_t有５个元素

        # Transition to self.end_tag
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.end_tag]]  # 其他标签到self.end_tag的转移概率
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.start_tag ]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # 11*5 经过了LSTM+Linear矩阵后的输出，之后作为CRF的输入。
        forward_score = self._forward_alg(feats)  # 0维的一个得分，20.*来着
        gold_score = self._score_sentence(feats, tags)  # tensor([ 4.5836])

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size, use_gpu):
    """初始化函数
    Args:
        vocab_size[int] : 单词数量
        embed_size[int] : 词向量维度
        use_gpu[boolean]: 是否使用GPU
    """
    super(SkipGramModel, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.input_embs = nn.Embedding(vocab_size, embed_size)
    self.input_embs.weight.data.uniform_(-1, 1)
    self.output_embs = nn.Embedding(vocab_size, embed_size)
    self.output_embs.weight.data.uniform_(-1, 1)
    if use_gpu:
        self.input_embs = self.input_embs.cuda()
        self.output_embs = self.output_embs.cuda() 


    def forward(self, pos_c, pos_v, neg_v):
    """前向传播.
    Args:
        pos_c[torch.tensor]: 中心词序列[batch_size,1]
        pos_v[torch.tensor]: 窗口词序列[batch_size, 1]
        neg_v[torch.tensor]: 负向词序列[batch_size, neg_size]
    Returns:
        Loss               : 损失值
    """
    emb_c = self.input_embs(pos_c)
    emb_v = self.output_embs(pos_v)
    score = torch.mul(emb_c, emb_v).sum(dim=-1)
    emb_v_neg = self.output_embs(neg_v)
    neg_score = torch.mul(emb_v_neg, emb_c).sum(dim=-1)
    pos_loss = F.logsigmoid(score).sum()
    neg_loss = F.logsigmoid(-1 * neg_score).sum()
    return -1 * (pos_loss + neg_loss)





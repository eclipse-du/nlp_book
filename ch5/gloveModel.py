import torch
import torch.nn as nn
import torch.nn.functional as F


class GloveModel(nn.Module):
    def __init__(self, vocab_size, embed_size, use_gpu):
        """初始化函数
        Args:
            vocab_size[int] : 单词数量
            embed_size[int] : 词向量维度
            use_gpu[boolean]: 是否使用GPU
        """
        super(GloveModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.input_embs = nn.Embedding(vocab_size, embed_size)
        self.input_embs.weight.data.uniform_(-1, 1)
        self.output_embs = nn.Embedding(vocab_size, embed_size)
        self.output_embs.weight.data.uniform_(-1, 1)
        if use_gpu:
            self.input_embs = self.input_embs.cuda()
            self.output_embs = self.output_embs.cuda() 


    def forward(self, pos_i, pos_j, wij, wf):
        """前向传播.
        Args:
            pos_i[torch.tensor]: 中心词序列[batch_size,1]
            pos_j[torch.tensor]: 窗口词序列[batch_size,1]
        Returns:
            Loss               : 损失值
        """
        batch_size = pos_i.shape[0]
        emb_i = self.input_embs(pos_i)
        emb_j = self.output_embs(pos_j)
        score = torch.mul(emb_i, emb_j).sum(dim=-1)
        bi = self.bi(pos_i)
        bj = self.bj(pos_j)
        nf = torch.pow((score.squeeze()+bi.squeeze() +
                        bj.squeeze() - torch.log(wij.squeeze())), 2)
        loss = (nf * wf).sum()
        return loss





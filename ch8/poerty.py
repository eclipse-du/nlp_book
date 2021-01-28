#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class TextConverter():
    def __init__(self,root):
        self.length = 870144
        self.root = root
        with open(self.root, 'r') as f:
            poetry_corpus = f.read()
        poetry_corpus = poetry_corpus.replace('\n', '')
        self.poetry_corpus = poetry_corpus[:self.length]
        vocab = set(poetry_corpus) #词典
        self.vocab = vocab
        self.step = 48
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    def build_text_array(self):
        text_array = []
        words = []
        for i in range(0,self.length,self.step):
            poem = self.poetry_corpus[i:self.step+i]#每次取一首诗，间隔为48
            for index in range(0,len(poem)):
                words.append(self.word_to_int(poem[index]))
            text_array.append(words)
            words=[]
        return np.array(text_array)
    
    def vocab_size(self):
        return len(self.vocab) + 1 #加1是因为需要增加未知词<unk>

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab) #因为返回最后一个unk

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')


# In[25]:


tc = TextConverter('./poetry.txt')
print(tc.build_text_array().shape)
print(tc.build_text_array()[0])


# In[3]:


from torch import nn
class PoetryModel(nn.Module):#charRNN
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 词向量层，词表大小 * 向量维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 网络主要结构
        self.rnn = nn.GRU(embedding_dim, self.hidden_dim,num_layers=2) #使用了GRU
        # 进行分类，分类的结果是字典中的一个字
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size() #可以查看当前Tensor的维度,返回的是seq_len以及batch_size分别是47和16
        #seq_len=47之后会解释
        if hidden is None:
            #self.directional*self.layer_num
            #区别在于 .data 返回和 x 的相同数据 tensor, 但不会加入到x的计算历史里，且require s_grad = False, 
            #这样有些时候是不安全的, 因为 x.data 不能被 autograd 追踪求微分 。
            #.detach() 返回相同数据的 tensor ,且 requires_grad=False ,但能通过 in-place 操作报告给 autograd 
            #在进行反向传播的时候.
            h_0 = input.detach().new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0= hidden
        # 输入 序列长度 * batch(每个汉字是一个数字下标)，
        # 输出 序列长度 * batch * 向量维度
        embeds = self.embeddings(input)
        output, hidden = self.rnn(embeds, h_0)
        output = self.linear(output.view(seq_len * batch_size, -1))#需要的就是seq_len和batch_size的乘积
        return output, hidden


# In[4]:


import torch as t
from torch.utils.data import DataLoader
from torch import optim

def train():
    device = t.device("cuda")
    # 获取数据
    tc = TextConverter('./poetry.txt')
    data = tc.build_text_array()
    data = t.from_numpy(data)
    dataloader = DataLoader(data,
                            batch_size=16,
                            shuffle=True,
                            num_workers=2)

    # 定义模型
    model = PoetryModel(tc.vocab_size(),
                        embedding_dim=256,
                        hidden_dim = 512)
    Configimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 转移到相应计算设备上
    model.to(device)
    # 进行训练
    for epoch in range(50):
        print('epoch='+str(epoch))
        for li,data_ in enumerate(dataloader):
            #data_的形状为torch.Size([16, 48])，代表的是batch以及一首诗
            data_ = data_.long().transpose(1,0)#是【48.16】
            data_ = data_.to(device)
            Configimizer.zero_grad()
            # n个句子，前n-1句作为输入，后n-1句作为输出，二者一一对应
            # 床前明月光，疑似地上霜
            # 锄禾日当午，汗滴禾下土
            #床前明月光，疑似地上
            #前明月光，疑似地上霜
            input_,target = data_[:-1,:],data_[1:,:]
            output,_ = model(input_)
            #print(output.shape) # [752,5387]其中752就是前47个字，target就是后47个字
            target = target.reshape(-1) #target的形状是[752]
            loss = criterion(output,target)
            loss.backward()
            Configimizer.step()
            if li % 1000 == 0:
                print('current loss = %.5f' % loss.item())
    t.save(model.state_dict(),'%s_%s.pth'%('./mypoem',epoch))
    print("finish training")


# In[5]:


train()


# In[6]:


# 藏头诗
def generate_acrostic(model, tc,start_words):
    results = list(start_words) #反清复明
    start_words_len = len(start_words)
    hidden=None
    poem=[]
    index =0
    input = t.Tensor([0]).view(1, 1).long()#其实应该是start
    input = input.cuda()
    for i in range(48):
        # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到最后的hidden
        if i % 12 == 0:
            w = results[index]
            poem.append(w) #反  清
            index += 1
            input = input.data.new([tc.word_to_int(w)]).view(1, 1)
        # 否则将output作为下一个input进行
        else:
            #top_index = output.data[0].topk(1)[1][0].item()
            _, top_index = t.max(output.data, 1) #返回字典里概率最大的字的index
            top_index = top_index.item()
            w = tc.int_to_word(top_index)
            poem.append(w)
            input = input.data.new([top_index]).view(1, 1)
        output,hidden = model(input,hidden)
    return poem


# In[20]:


tc = TextConverter('./poetry.txt')
model = PoetryModel(tc.vocab_size(),256,512)
model.load_state_dict(t.load('./mypoem_49.pth', 'cpu'))
model.cuda()
print(''.join(generate_acrostic(model,tc, '厨娘无敌')))


# In[ ]:





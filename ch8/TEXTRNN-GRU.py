#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
from torchtext.data import Field

#定义分词方法
def tokenizer(text): # create a tokenizer function
    stopwords = stopwordslist('/root/news/stopword.txt')  # 这里加载停用词的路径  
    return [word for word in jieba.cut(text) if word.strip() not in stopwords] #使用jieba做中文分词并且加载停用词

#加载停用词词库
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()]  
    return stopwords 

#Field类处理数据
TEXT = Field(sequential=True, tokenize=tokenizer, fix_length=200) #使用了分词方法tokenizer
LABEL = Field(sequential=False)

tv_datafields = [("text",TEXT),("label",LABEL)]


# In[2]:


import re
import os

def CreateDataSet(root):
    # 定义正则表达式
    patternUrl = re.compile(r'<url>(.*?)</url>', re.S)
    patternContent = re.compile(r'<content>(.*?)</content>', re.S)
    contents_list = []
    classes_list = []
    #看看这些新闻的种类共有多少以及每个种类有多少篇新闻：
    for file in os.listdir(root):
        # 设置路径打开文件
        file_path = os.path.join(root, file)
        with open(file_path,'r') as f:
            text = f.read()
            # 正则匹配出url和content
            urls = patternUrl.findall(text)
            contents = patternContent.findall(text)
            for i in range(urls.__len__()):
                patternClass = re.compile(r'http://(.*?).sohu.com', re.S) #http://yule.sohu.com
                classes_list.append(patternClass.findall(urls[i])[0])
                contents_list.append(contents[i])
    return classes_list,contents_list


# In[3]:


import time
start_time=time.time()
classes_list,contents_list = CreateDataSet("/root/news/sogu")
end_time=time.time()
print('数据源长度为 '+str(len(classes_list)))
print("读入文件的总时间为: "+str(end_time-start_time))


# In[4]:


# 去除几个不需要的种类，同时删除字数小于100字的新闻
for i in range(contents_list.__len__())[::-1]:
    if (len(contents_list[i]) < 100 or classes_list[i] == '2008'
            or classes_list[i] == 'cul' or classes_list[i] == 'mil.news' or classes_list[i] == 'career'):
        contents_list.pop(i)
        classes_list.pop(i)


# In[5]:


from collections import Counter
Counter(classes_list)


# In[6]:


# 每一类提取3000个新闻
X = []
y = []
d = {"business":0, "health":0, "house":0, "it":0, 
     "learning":0, "news":0, "sports":0, "travel":0, "women":0, "yule":0,"auto":0}
for i in range(len(classes_list)):
    if (d[classes_list[i]] < 3000):
        d[classes_list[i]] += 1
        X.append(contents_list[i])
        y.append(classes_list[i])


# In[7]:


from concurrent.futures import ProcessPoolExecutor
from torchtext import data, datasets
import time
start1=time.time()
future_list =[]
datasets = []
datagroup = zip(X,y) #组成tuple
def constructDataset(dataset):
    return data.Example.fromlist([dataset[0], dataset[1]], tv_datafields)

with ProcessPoolExecutor(16) as executor:
    for dataset in datagroup:
        future = executor.submit(constructDataset,dataset)
        future_list.append(future)

for f in future_list:
    datasets.append(f.result())
    
end1=time.time()
print("consturct dataset consumes  : "+str(end1-start1))
print('构建dataset之后的数据长度为 '+ str(len(datasets)))


# In[8]:


import random
random.shuffle(datasets)
traintest_index = int(0.8*len(datasets))
trainset,testset = datasets[:traintest_index],datasets[traintest_index:]
print('training set= ' + str(len(trainset)))
print('testing set = ' + str(len(testset)))


# In[9]:


from torchtext.vocab import Vectors
import torch
def get_data_iter(text_field, label_field, train_data,test_data,batch_size):
    vectors=Vectors(name='/root/news/sgns.sogou.word') #使用预训练的词向量，维度为300Dimension
    text_field.build_vocab(train_data, vectors=vectors)
    label_field.build_vocab(train_data)
    # print vocab information
    print('len(TEXT.vocab)', len(text_field.vocab))
    print('TEXT.vocab.vectors.size()', text_field.vocab.vectors.size())
    print('len(label_field.vocab):',len(label_field.vocab))
    train_iter, test_iter = data.Iterator.splits(
            (train_data, test_data),
            batch_sizes=(batch_size,batch_size),
        device=torch.device
        ("cuda:0" if torch.cuda.is_available() else "cpu"),repeat=False,sort=False)
        #device = torch.device("cpu"),repeat=False,sort=False)
    return train_iter, test_iter


# In[10]:


from torchtext import data, datasets
batch_size = 64
traindataset = data.Dataset(examples=trainset,fields=tv_datafields)
testdataset = data.Dataset(examples=testset,fields=tv_datafields)
train_iter,test_iter = get_data_iter(TEXT,LABEL,traindataset,testdataset,batch_size)
vocab_size = len(TEXT.vocab)
label_size = len(LABEL.vocab)


# In[12]:


import torch.nn as nn
import torch
class MyLSTM(nn.Module):
    def __init__(self, vocab_size,embed_size,hidden_size, out_size, 
                 n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, #词向量的总长度
                                embedding_dim=embed_size)  # 创建词向量对象,embedding_size为词向量的维度

        # 这里指定了 BATCH FIRST
        self.rnn = torch.nn.GRU(embed_size, hidden_size, 
                                n_layers,batch_first=True)

       # 加了一个线性层，全连接
        self.out = torch.nn.Linear(hidden_size, out_size)
        #self.hidden = self.init_hidden
        
    def forward(self,x,h_state):
        #获得词嵌入矩阵
        embeds = self.embedding(x)#torch.Size([64, 200, 300])
        #output里包含了h1-ht所有的输出，hidden的最后结果输出就是类别
        output, hidden = self.rnn(embeds, h_state)
        output = self.out(hidden)
        return output
    
    #def init_hidden(self):
        #return torch.zeros(1, 64, hidden_size)
    
model = MyLSTM(vocab_size,300,128,18,1)
model.cuda()


# In[14]:


learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
criterion = nn.CrossEntropyLoss()

num_epoches = 100
model.train()
for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for batch_idx, (data,target) in enumerate(train_iter):
        contents,labels = data[0].cuda(),data[1].cuda()
        contents = contents.permute(1,0) #将contents的尺寸变为(batchsize,新闻文章的最长长度)
        batch_size_in= contents.shape[0]
        h_state = torch.zeros(1, batch_size_in, 128).cuda()
        output = model(contents,h_state) #将数据集传入网络做前向计算
        labels = labels-1
        output = output[-1,:,:]
        loss = criterion(output, labels) #计算loss
        optimizer.zero_grad() #在做反向传播之前先清除下网络状态
        loss.backward() #loss反向传播
        optimizer.step() #更新参数
        if batch_idx % 100 == 0:
            print('current loss = %.5f' % loss.item())
print("finish training")


# In[15]:


# 做 prediction
total = 0
correct = 0
model.eval()
for data, target in test_iter:
    contents,labels = data[0].cuda(),data[1].cuda()
    contents = contents.permute(1,0) #变为batchsize,vocab_len
    batch_size_in= contents.shape[0]
    h_state = torch.zeros(1, batch_size_in, 128).cuda()
    output = model(contents,h_state) #将数据集传入网络做前向计算
    labels = labels-1
    output = output[-1,:,:]
    #hidden = hidden.view(-1,150)
    _, predicts = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).cpu().sum()
print('Accuracy = %.2f' % (100 * correct / total))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
from torchtext.data import Field

#定义分词方法
def tokenizer(text): # create a tokenizer function
    stopwords = stopwordslist('./stopwords.txt')  # 这里加载停用词的路径  
    return [word for word in jieba.cut(text) if word.strip() not in stopwords] #使用jieba做中文分词并且加载停用词

#加载停用词词库
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()]  
    return stopwords 

#Field类处理数据
TEXT = Field(sequential=True, tokenize=tokenizer, batch_first=True,fix_length=200) #使用了分词方法tokenizer
LABEL = Field(sequential=False)


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
        with open(file_path,'r',encoding='gb18030') as f:
            text = f.read()
            # 正则匹配出url和content
            urls = patternUrl.findall(text)
            contents = patternContent.findall(text)
            for i in range(urls.__len__()):
                patternClass = re.compile(r'http://(.*?).sohu.com', re.S) #http://yule.sohu.com
                classes_list.append(patternClass.findall(urls[i])[0])
                contents_list.append(contents[i])
    return classes_list,contents_list


# In[4]:


import time
start_time=time.time()
classes_list,contents_list = CreateDataSet("./news/")
end_time=time.time()
print('数据源长度为 '+str(len(classes_list)))
print("读入文件的总时间为: "+str(end_time-start_time))


# In[8]:


# 去除几个不需要的种类，同时删除字数小于100字的新闻
for i in range(contents_list.__len__())[::-1]:
    if (len(contents_list[i]) < 100 or classes_list[i] == '2008'
            or classes_list[i] == 'cul' or classes_list[i] == 'mil.news' or classes_list[i] == 'career'):
        contents_list.pop(i)
        classes_list.pop(i)


# In[9]:


from collections import Counter
Counter(classes_list)


# In[10]:


# 每一类提取2000个新闻
X = []
y = []
d = {"business":0, "health":0, "house":0, "it":0, 
     "learning":0, "news":0, "sports":0, "travel":0, "women":0, "yule":0,"auto":0}
for i in range(len(classes_list)):
    if (d[classes_list[i]] < 2000):
        d[classes_list[i]] += 1
        X.append(contents_list[i])
        y.append(classes_list[i])


# In[11]:


import random
datagroup = list(zip(X,y)) #组成tuple
length = len(datagroup)
random.shuffle(datagroup)
trainvalid_index = int(0.8*length)
trainset,validset = datagroup[:trainvalid_index],datagroup[trainvalid_index:]


# In[12]:


from torchtext import data
from tqdm import tqdm
import numpy as np
import random

#定义Dataset
class MyDataset(data.Dataset):

    def __init__(self,datatuple,text_field,label_field,test=False):
        fields = [("text",text_field),("label",label_field)]      
        examples = []
        if test:
            # 如果为测试集，则不加载label
            for content,label in tqdm(datatuple):
                examples.append(data.Example.fromlist([content, None], fields))
        else:
            for content, label in tqdm(datatuple):
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(data.Example.fromlist([content,label], fields))
        # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类。
        super().__init__(examples, fields)


# In[13]:


train=MyDataset(trainset,text_field=TEXT,label_field=LABEL,test=False)
valid=MyDataset(validset,text_field=TEXT,label_field=LABEL,test=False)


# In[14]:


from torchtext.vocab import Vectors
vectors=Vectors(name='./sgns.sogou.word') #使用预训练的词向量，维度为300Dimension
TEXT.build_vocab(train, vectors=vectors)
LABEL.build_vocab(train)


# In[15]:


import torch
from torchtext.data import BucketIterator
batchsize=64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter = data.BucketIterator(dataset=train, batch_size=batchsize, 
        shuffle=True, sort_key=lambda x: len(x.text),
        device=DEVICE,sort_within_batch=False, repeat=False)
valid_iter = data.BucketIterator(dataset=valid, batch_size=batchsize, 
        shuffle=True, sort_key=lambda x: len(x.text),
        device=DEVICE,sort_within_batch=False, repeat=False)


# In[16]:


vocab_size = len(TEXT.vocab)
print(vocab_size)
vocab_vectors = TEXT.vocab.vectors.numpy()


# In[17]:


print(LABEL.vocab.stoi)


# In[18]:


import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, num_class,vocab_size,embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.embedding = nn.Embedding(num_embeddings=vocab_size, #词向量的总长度
                                embedding_dim=embedding_size)  # 创建词向量对象,embedding_size为词向量的维度
        #self.embedding.weight.data.copy_(torch.from_numpy(vocab_vectors))
        #self.embedding.weight.requires_grad = False #使用的是静态词向量
     
        filter_num = 128 # 每种卷积核的数量
        Ks = [2,3,4] #卷积核list，形如[2,3,4]
        
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embedding_size, 
                    filter_num, 
                    kernel_size=k),  #
                    nn.ReLU(), # 激活函数层
                    nn.MaxPool1d(kernel_size=200-k+1))          
                    for k in Ks
                    ])

        self.dropout = nn.Dropout() 
        self.fc = nn.Linear(len(Ks)*filter_num,num_class,bias=True) ##全连接层

    
    def forward(self, x):
        output = self.embedding(x)
        #torch.Size([64, 200, 300])
        # 输出尺寸为batch_size x text_len x embedding_size 
        #需要转为batch_size x embedding_size x text_len
        output = output.permute(0,2,1)#一维卷积是在最后维度上扫描，所以需要改变形状
        output = [conv(output) for conv in self.convs]
        #out[i]:batch_size x feature_size x 1
        output = torch.cat(output, dim=1)#对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        output = output.view(output.size(0), -1)#取之前输出的第一维，其他flatten
        output = self.dropout(output)
        output = self.fc(output)
        return output


# In[ ]:


model = TextCNN(12,vocab_size,300)
model.cuda()
learning_rate = 1e-3
print(model.embedding.weight.data) #随机初始化的词向量
print('---------')
model.embedding.weight.data.copy_(torch.from_numpy(vocab_vectors)) #使用了预训练词向量
model.embedding.weight.requires_grad = False
print(model.embedding.weight.data)
#这个就是静态加载词向量，使之后续不需要被更新
optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad==True], lr=learning_rate) 
criterion = nn.CrossEntropyLoss().cuda()


# In[19]:


#model = TextCNN(12,vocab_size,300)
#model.cuda()
#learning_rate = 1e-3
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
#criterion = nn.CrossEntropyLoss()
num_epoches = 50
model.train()
for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for batch_idx, (data,target) in enumerate(train_iter):
        contents,labels = data[0].cuda(),data[1].cuda()
        #contents = contents.permute(1,0) #将contents的尺寸变为(batchsize,新闻文章的最长长度)
        outputs = model(contents) #将数据集传入网络做前向计算
        loss = criterion(outputs, labels) #计算loss
        optimizer.zero_grad() #在做反向传播之前先清除下网络状态
        loss.backward() #loss反向传播
        optimizer.step() #更新参数
        
        if batch_idx % 10000 == 0:
            print('current loss = %.5f' % loss.item())
print("finish training")


# In[21]:


# 做 prediction
total = 0
correct = 0
model.eval()
for data, labels in valid_iter:
    contents,labels = data[0].cuda(),data[1].cuda()
    #contents = contents.permute(1,0) #变为batchsize,vocab_len
    outputs = model(contents)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).cpu().sum()

print('Accuracy = %.2f' % (100 * correct // total))


# In[ ]:


print(model.embedding.weight.data)


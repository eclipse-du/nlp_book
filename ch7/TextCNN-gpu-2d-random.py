#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
from torchtext.data import Field

#定义分词方法
def tokenizer(text): # create a tokenizer function
    stopwords = stopwordslist('./stopword.txt')  # 这里加载停用词的路径  
    return [word for word in jieba.cut(text) if word.strip() not in stopwords] #使用jieba做中文分词并且加载停用词

#加载停用词词库
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()]  
    return stopwords 

#Field类处理数据，但是在RNN模型中batch_first一定不要设置为first
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


# In[3]:


import time
start_time=time.time()
classes_list,contents_list = CreateDataSet("./news")
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


# In[7]:


import random
datagroup = list(zip(X,y)) #组成tuple
length = len(datagroup)
random.shuffle(datagroup) #打散数据
trainvalid_index = int(0.8*length)
trainset,validset = datagroup[:trainvalid_index],datagroup[trainvalid_index:]


# In[8]:


from torchtext import data
from tqdm import tqdm
import numpy as np
import random

#定义Dataset
class MyDataset(data.Dataset):

    def __init__(self,datatuple,text_field,label_field,test=False):
        fields = [("text",text_field),("label",label_field)]      
        datasets = [] #因为examples数组的名字和torchtext中data.Example太接近了
        if test:
            # 如果为测试集，则不加载label
            for content,label in tqdm(datatuple):
                datasets.append(data.Example.fromlist([content, None], fields))
        else:
            for content, label in tqdm(datatuple):
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                datasets.append(data.Example.fromlist([content,label], fields))
        # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类。
        super().__init__(datasets, fields)


# In[9]:


train=MyDataset(trainset,text_field=TEXT,label_field=LABEL,test=False)
valid=MyDataset(validset,text_field=TEXT,label_field=LABEL,test=False)


# In[10]:


from torchtext.vocab import Vectors
#vectors=Vectors(name='/root/news/sgns.sogou.word') #使用预训练的词向量，维度为300Dimension
TEXT.build_vocab(train) #没有使用预训练的词向量,通过build_vocab方法来构建训练集的字典
LABEL.build_vocab(train)


# In[11]:


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


# In[12]:


vocab_size = len(TEXT.vocab)
print(vocab_size)
#vocab_vectors = TEXT.vocab.vectors.numpy()


# In[13]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, num_class,vocab_size,embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.embedding = nn.Embedding(num_embeddings=vocab_size, #词向量的总长度
                         embedding_dim=embedding_size)  # 创建词向量对象,embedding_size为词向量的维度
        #self.embedding.weight.data.copy_(torch.from_numpy(vocab_vectors))
        self.embedding.weight.requires_grad = True #更新词向量
        channel = 1 #输入的channel数
        filter_num = 128 # 每种卷积核的数量
        Ks = [2,3,4] # 卷积核list，形如[2,3,4]
        self.convs = nn.ModuleList([nn.Conv2d(channel,filter_num,(K,embedding_size)) for K in Ks]) # 卷积层
        self.dropout = nn.Dropout()
        #对于输入图为何是len(ks)*filter_num看textcnn的论文图就知道了。
        self.fc = nn.Linear(len(Ks)*filter_num,num_class,bias=True) #全连接层
    
    def forward(self, x):
        # batch_size x 1 × max_sent_len x embedding_dim
        output = self.embedding(x) #torch.Size([64, 200, 300])
        output = output.unsqueeze(1)#torch.Size([64, 1, 200, 300])，参看conv2d的输入数据尺寸
        output = [conv(output) for conv in self.convs]
        #torch.Size([64, 128, 199, 1])
        # 经过最大池化层,维度变为(batch_size, out_chanel, h=1, w=1)
        output = [F.max_pool2d(input=x_item, 
                kernel_size=(x_item.size(2), x_item.size(3))) for x_item in output]
        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        output = [x_item.view(x_item.size(0), -1) for x_item in output]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        output = torch.cat(output, 1)
        # dropout层
        output = self.dropout(output)

        # 全连接层
        output = self.fc(output)
        return output       


# In[14]:


model = TextCNN(12,vocab_size,300)
model.to(DEVICE)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
criterion = nn.CrossEntropyLoss().to(DEVICE)
num_epoches = 20
model.train()
for epoch in range(num_epoches):
    print('current epoch = %d' % epoch)
    for batch_idx, (data,target) in enumerate(train_iter):
        contents,labels = data[0].to(DEVICE),data[1].to(DEVICE)
        #contents = contents.permute(1,0) #将contents的尺寸变为(batchsize,新闻文章的最长长度)
        outputs = model(contents) #将数据集传入网络做前向计算
        loss = criterion(outputs, labels) #计算loss
        optimizer.zero_grad() #在做反向传播之前先清除下网络状态
        loss.backward() #loss反向传播
        optimizer.step() #更新参数
        
        if batch_idx % 10000 == 0:
            print('current loss = %.5f' % loss.item())
print("finish training")


# In[ ]:


# 做 prediction
total = 0
correct = 0
model.eval()
for data, labels in valid_iter:
    contents,labels = data[0].to(DEVICE),data[1].to(DEVICE)
    #contents = contents.permute(1,0) #变为batchsize,vocab_len
    outputs = model(contents)
    _, predicts = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicts == labels).cpu().sum()

print('Accuracy = %.2f' % (100 * correct // total))


# In[ ]:





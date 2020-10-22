"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/8/24 16:33
"""
import numpy as np
import os
import jieba
import json
import torch
import pickle as pkl
from torch.utils.data import Dataset


def get_seg(sentence):
    """
    对一句话进行分词操作,使用jieba作为工具
    :param sentence [str]: 输入为一句完整的句子
    :return: sentence_seg [list]: 内部每一个元素为词语
    """
    sentence_seg = jieba.cut(sentence.strip())
    sentence_seg = " ".join(sentence_seg)
    sentence_seg = sentence_seg.split(" ")
    return sentence_seg


def load_vocab(char_dim, char_dictionary, char_vactor_file):
    """
    导入预先训练好的词向量,并判断其词典是否存在，如果不存在则生成对应词典
    :param char_dim [int]: 词向量维度大小
    :param char_dictionary [str]: 词典路径
    :param char_vactor_file [str]: 词向量路径
    :return:
    """
    vec_size = char_dim
    if os.path.exists(char_dictionary) is False:
        # 如果词典不存在，先生成词典
        vocabulary = {}
        with open(char_vactor_file, "r", encoding="utf-8") as fh:
            i = 2
            for line in fh.readlines():
                array = line.rstrip().split(" ")
                char = "".join(array[0:-vec_size])
                vocabulary[char] = i
                i += 1
        vocabulary['OOV'] = int(0)
        vocabulary['UNK'] = int(1)
        # 将词典写入文件中
        write_txt(vocabulary, char_dictionary)
    # 读入词典文件，生成词典
    vocabulary = read_dictionary(char_dictionary)
    embedding_dict = {}
    # 生成词向量，并在其中加入未登录词UNK词向量，以及补充长度OOV词向量
    with open(char_vactor_file, "r", encoding="utf-8") as fh:
        for line in fh.readlines():
            array = line.rstrip().split(" ")
            char = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            embedding_dict[char] = vector
    embedding_dict['OOV'] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    embedding_dict['UNK'] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in vocabulary.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return vocabulary, emb_mat


def write_txt(data, filename):
    # 将字典或词典存到txt文件中
    f = open(filename, "w", encoding="utf-8")
    if type(data) == dict:
        new_dic = sorted(dict2list(data), key=lambda x: x[1], reverse=False)
        for term in new_dic:
            templine = "%s\t%d\n" % (term[0], term[1])
            f.write(templine)
        f.close()


def dict2list(dic: dict):
    # 对词典进行排序
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst


def read_dictionary(char_dictionary):
    """
    读入已经存好的词典
    :param char_dictionary [str]: 词典路径
    :return: dict词典
    """
    with open(char_dictionary, "r", encoding="utf-8") as fh:
        vocabulary = {}
        for i, str_ in enumerate(fh.readlines()):
            s_list = str_.rstrip("\n").split("\t")
            vocabulary[s_list[0]] = int(s_list[1])
    return vocabulary


class ChoiceDatum:
    """
    构建答案导向任务数据类
    raw_text1[str]:用户回答文本
    raw_text2[str]:表示选项文本
    label[int]：0为不选择该选项，1为选择该选项
    question_id[str]:问题的id
    """
    def __init__(self, raw_text1, raw_text2, label, question_id):
        self.raw_text1 = raw_text1
        self.raw_text2 = raw_text2
        self.label = label
        self.question_id = question_id

    def set_word_idxs(self, contentA_idxs, contentB_idxs):
        self.contentA_idxs = contentA_idxs
        self.contentB_idxs = contentB_idxs

    def set_sentence_len(self, contentA_len, contentB_len):
        self.contentA_len = contentA_len
        self.contentB_len = contentB_len


def construct_dataset(dataset_dir):
    """
    通过文件路径，构造需要的数据集，并进行分词
    :param dataset_dir [str]: 原始答案导向任务数据路径
    :return: list类型数据，其中每个元素为ChoiceDatum类型数据
    """
    output = []
    with open(dataset_dir, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh.readlines():
            # sample = json.loads(line.strip())
            sample = eval(line.strip())
            ids = sample["ids"]
            content = sample["passage"]
            answer = sample["answer"]
            alternatives = sample["alternatives"]
            alterTemp = alternatives.split("|")
            answerTemp = answer.split("|")
            for alter in alterTemp:
                idsTemp = ids
                if alter in answerTemp:
                    label = 1
                else:
                    label = 0
                contentA = get_seg(content)
                contentB = get_seg(alter)
                output.append(ChoiceDatum(raw_text1=contentA, raw_text2=contentB,
                                          label=label, question_id=idsTemp))
    return output


def convert_features(contentA, contentB, word2idx_dict, max_length_content):
    """
    根据词典，将分好词的句子中token转化成dict_id
    :param contentA [list]: 分过词的用户回答文本
    :param contentB [list]: 分过词的选项文本
    :param word2idx_dict [dict]: 词典
    :param max_length_content [int]: 规定文本的最大长度
    :return: 两个文本dict_id的list，以及对应的真实长度
    """
    max_length_contentA = max_length_content
    max_length_contentB = max_length_content
    if len(contentA) > max_length_contentA:
        contentA = contentA[:int(max_length_contentA)]
    else:
        contentA = contentA
    contentA_len = len(contentA)
    if len(contentB) > max_length_contentB:
        contentB = contentB[:int(max_length_contentB)]
    else:
        contentB = contentB
    contentB_len = len(contentB)
    contentA_idxs = np.zeros([max_length_contentA], dtype=np.int32)
    contentB_idxs = np.zeros([max_length_contentB], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1
    for i, token in enumerate(contentA):
        contentA_idxs[i] = _get_word(token)
    for i, token in enumerate(contentB):
        contentB_idxs[i] = _get_word(token)
    return contentA_idxs, contentB_idxs, contentA_len, contentB_len


def process_text_dataset(dataset, word2idx_dict, max_length_content):
    """
    将token转化成dict_id的数据，加入到构建答案导向任务数据类中
    :param dataset: list类型数据，其中每个元素为ChoiceDatum类型数据
    :param word2idx_dict: 词典
    :param max_length_content: 规定文本的最大长度
    :return: list类型数据，其中每个元素为ChoiceDatum类型数据
    """
    for i in range(len(dataset)):
        contentA = dataset[i].raw_text1
        contentB = dataset[i].raw_text2
        contentA_idxs, contentB_idxs, contentA_len, contentB_len = \
            convert_features(contentA, contentB, word2idx_dict, max_length_content)
        dataset[i].set_word_idxs(contentA_idxs, contentB_idxs)
        dataset[i].set_sentence_len(contentA_len, contentB_len)
    return dataset


class ChoiceDataset(Dataset):
    """
    创建答案导向任务数据集类
    """
    def __init__(self, data_list, max_length_content):
        # data_list: list of ChoiceDatum
        self.data_list = data_list
        self.max_length_content = max_length_content

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        contentA_idxs, contentB_idxs, contentA_len, contentB_len, label = self.data_list[key].contentA_idxs, \
                                                                          self.data_list[key].contentB_idxs, \
                                                                          self.data_list[key].contentA_len, \
                                                                          self.data_list[key].contentB_len, \
                                                                          self.data_list[key].label
        return contentA_idxs, contentB_idxs, contentA_len, contentB_len, label, self.max_length_content


def choice_collate_func(batch):
    """
    构建batch生成函数，将并numpy类型的数据转化成torch所需的类型
    :param batch [list]: batch_size大小个数据
    :return:
    """
    data_list1 = []
    data_list2 = []
    label_list = []
    sentence_length_list1 = []
    sentence_length_list2 = []
    for datum in batch:
        data_list1.append(datum[0])
        data_list2.append(datum[1])
        sentence_length_list1.append(datum[2])
        sentence_length_list2.append(datum[3])
        label_list.append(datum[4])
    return [torch.LongTensor(np.array(data_list1)), torch.LongTensor(np.array(data_list2)),
            torch.LongTensor(sentence_length_list1), torch.LongTensor(sentence_length_list2),
            torch.LongTensor(label_list)]


def save_data(filename, obj, message=None):
    if message is not None:
        print("Save {}...".format(message))
        with open(filename, "wb") as fh:
            pkl.dump(obj, fh)


def save_emb(filename, obj, message=None):
    if message is not None:
        print("Save {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    """
    config.word_dims[int]: 词向量维度
    config.word_dictionary[str]: 词典路径
    config.word_vector_file[str]: 预训练词向量路径
    config.train_file[str]: 原始训练集路径
    config.train_eval[str]: 生成的训练集ChoiceDatum类集合保存路径
    config.dev_file[str]: 原始验证集路径
    config.dev_eval[str]: 生成的验证集ChoiceDatum类集合保存路径
    config.word_emb_file[str]: 词向量保存路径
    """
    word2idx_dict, word_mat = load_vocab(config.word_dims, config.word_dictionary, config.word_vector_file)
    train_set = construct_dataset(config.train_file)
    dev_set = construct_dataset(config.dev_file)
    train_data = process_text_dataset(train_set, word2idx_dict, config.max_length_content)
    dev_data = process_text_dataset(dev_set, word2idx_dict, config.max_length_content)
    save_data(config.train_eval, train_data, message="train data")
    save_data(config.dev_eval, dev_data, message="dev data")
    if os.path.exists(config.word_emb_file) is False:
        save_emb(config.word_emb_file, word_mat, message="word embedding")


def data_loader(path_dir, batch_size, max_length_content, shuffle=True):
    """
    生成batch_size大小的数据
    :param path_dir [str]: 答案导向任务数据类的数据路径
    :param batch_size [int]: 批次输入到模型中样本数量的大小
    :param max_length_content [int]: 规定文本的最大长度
    :param shuffle [bool]: 是否将数据集的顺寻打乱
    :return: batch_size大小的loader
    """
    with open(path_dir, "rb") as fh:
        data = pkl.load(fh)
    choice_data = ChoiceDataset(data, max_length_content)
    # construct data loader
    choice_data_loader = torch.utils.data.DataLoader(dataset=choice_data,
                                                     batch_size=batch_size,
                                                     collate_fn=choice_collate_func,
                                                     shuffle=shuffle)
    return choice_data_loader, len(choice_data)


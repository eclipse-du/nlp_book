"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/8/31 14:46
"""
import numpy as np
import os
import jieba
import json
import torch
import pickle as pkl
from torch.utils.data import Dataset


def get_seg(sentence):
    # 对一句话，用jieba分词
    sentence_seg = jieba.cut(sentence.strip())
    sentence_seg = " ".join(sentence_seg)
    sentence_seg = sentence_seg.split(" ")
    return sentence_seg


def load_vocab(char_dim, char_dictionary, char_vactor_file):
    # 导入字或词向量，判断字典或词典是否存在，如果不存在则生成字典或词典
    vec_size = char_dim
    if os.path.exists(char_dictionary) is False:
        vocabulary = {}
        with open(char_vactor_file, "r", encoding="utf-8") as fh:
            i = 2
            for line in fh.readlines():
                array = line.rstrip().split(" ")
                char = "".join(array[0:-vec_size])
                # if char == "":
                #     char = "<%d>"%i
                vocabulary[char] = i
                i += 1
        vocabulary['OOV'] = int(0)
        vocabulary['UNK'] = int(1)
        write_txt(vocabulary, char_dictionary)

    vocabulary = read_dictionary(char_dictionary)
    embedding_dict = {}
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
    # 对字典进行排序
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst


def read_dictionary(char_dictionary):
    # 读入已经存好的字典或词典
    with open(char_dictionary, "r", encoding="utf-8") as fh:
        vocabulary = {}
        for i, str_ in enumerate(fh.readlines()):
            s_list = str_.rstrip("\n").split("\t")
            vocabulary[s_list[0]] = int(s_list[1])
    return vocabulary


class ChoiceDatum:
    # 构建选择题数据类
    # raw_text1表示用户回答文本， raw_text2表示选项文本
    # label：0为不选择该选项，1为选择该选项
    # question_id表示问题的id
    def __init__(self, raw_text1, raw_text2, label, question_id):
        self.raw_text1 = raw_text1
        self.raw_text2 = raw_text2
        self.label = label
        self.question_id = question_id

    def set_word_idxs(self, contentA_idxs, contentB_idxs):
        self.contentA_idxs = contentA_idxs
        self.contentB_idxs = contentB_idxs

    def set_char_idxs(self, contentA_char_idxs, contentB_char_idxs):
        self.contentA_char_idxs = contentA_char_idxs
        self.contentB_char_idxs = contentB_char_idxs


def construct_dataset(dataset_dir):
    # 通过文件路径，构造需要的数据集，并进行分词
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
                    for _ in range(2):
                        contentA = get_seg(content)
                        contentB = get_seg(alter)
                        output.append(ChoiceDatum(raw_text1=contentA, raw_text2=contentB,
                                                  label=label, question_id=idsTemp))
                else:
                    label = 0
                    contentA = get_seg(content)
                    contentB = get_seg(alter)
                    output.append(ChoiceDatum(raw_text1=contentA, raw_text2=contentB,
                                              label=label, question_id=idsTemp))
    return output


def convert_features(contentA, contentB, word2idx_dict, char2idx_dict, max_length_content, char_limit):
    # 根据词典，将分好词的句子中token转化成dict_id
    max_length_contentA = max_length_content
    max_length_contentB = max_length_content
    if len(contentA) > max_length_contentA:
        contentA = contentA[:int(max_length_contentA)]
    else:
        contentA = contentA
    if len(contentB) > max_length_contentB:
        contentB = contentB[:int(max_length_contentB)]
    else:
        contentB = contentB
    contentA_idxs = np.zeros([max_length_contentA], dtype=np.int32)
    contentB_idxs = np.zeros([max_length_contentB], dtype=np.int32)
    contentA_char_idxs = np.zeros([max_length_contentA, char_limit], dtype=np.int32)
    contentB_char_idxs = np.zeros([max_length_contentB, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(contentA):
        contentA_idxs[i] = _get_word(token)
    for i, token in enumerate(contentB):
        contentB_idxs[i] = _get_word(token)

    for i, token in enumerate(contentA):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            contentA_char_idxs[i, j] = _get_char(char)
    for i, token in enumerate(contentB):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            contentB_char_idxs[i, j] = _get_char(char)
    return contentA_idxs, contentB_idxs, contentA_char_idxs, contentB_char_idxs


def process_text_dataset(dataset, word2idx_dict, char2idx_dict, max_length_content, char_limit):
    # 将token转化成dict_id的数据，加入到选择题数据类中
    for i in range(len(dataset)):
        contentA = dataset[i].raw_text1
        contentB = dataset[i].raw_text2
        contentA_idxs, contentB_idxs, contentA_char_idxs, contentB_char_idxs = \
            convert_features(contentA, contentB, word2idx_dict, char2idx_dict, max_length_content, char_limit)
        dataset[i].set_word_idxs(contentA_idxs, contentB_idxs)
        dataset[i].set_char_idxs(contentA_char_idxs, contentB_char_idxs)
    return dataset


class ChoiceDataset(Dataset):
    # 创建选择题数据集类
    def __init__(self, data_list, max_length_content):
        # data_list: list of ChoiceDatum
        self.data_list = data_list
        self.max_length_content = max_length_content

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        contentA_idxs, contentB_idxs, contentA_char_idxs, contentB_char_idxs, label = self.data_list[key].contentA_idxs, \
                                                                                    self.data_list[key].contentB_idxs, \
                                                                                    self.data_list[key].contentA_char_idxs, \
                                                                                    self.data_list[key].contentB_char_idxs, \
                                                                                    self.data_list[key].label
        return contentA_idxs, contentB_idxs, contentA_char_idxs, contentB_char_idxs, label, self.max_length_content


def choice_collate_func(batch):
    data_list1 = []
    data_list2 = []
    label_list = []
    data_char_list1 = []
    data_char_list2 = []
    for datum in batch:
        data_list1.append(datum[0])
        data_list2.append(datum[1])
        data_char_list1.append(datum[2])
        data_char_list2.append(datum[3])
        label_list.append(datum[4])
    return [torch.LongTensor(np.array(data_list1)), torch.LongTensor(np.array(data_list2)),
            torch.LongTensor(np.array(data_char_list1)), torch.LongTensor(np.array(data_char_list2)),
            torch.LongTensor(label_list)]


def save(filename, obj, message=None):
    if message is not None:
        print("Save {}...".format(message))
        with open(filename, "wb") as fh:
            pkl.dump(obj, fh)


def save2(filename, obj, message=None):
    if message is not None:
        print("Save {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    word2idx_dict, word_mat = load_vocab(config.word_dims, config.word_dictionary, config.word_vector_file)
    char2idx_dict, char_mat = load_vocab(config.char_dims, config.char_dictionary, config.char_vector_file)
    train_set = construct_dataset(config.train_file)
    dev_set = construct_dataset(config.dev_file)
    train_data = process_text_dataset(train_set, word2idx_dict, char2idx_dict, config.max_length_content, config.char_limit)
    dev_data = process_text_dataset(dev_set, word2idx_dict, char2idx_dict, config.max_length_content, config.char_limit)
    save(config.train_eval, train_data, message="train data")
    save(config.dev_eval, dev_data, message="dev data")
    if os.path.exists(config.word_emb_file) is False:
        save2(config.word_emb_file, word_mat, message="word embedding")
    if os.path.exists(config.char_emb_file) is False:
        save2(config.char_emb_file, char_mat, message="char embedding")


def data_loader(path_dir, batch_size, max_length_content, shuffle=True):
    print("load train_eval ...")
    with open(path_dir, "rb") as fh:
        data = pkl.load(fh)
    # data = eval(data_)
    choice_data = ChoiceDataset(data, max_length_content)
    # construct data loader
    choice_data_loader = torch.utils.data.DataLoader(dataset=choice_data,
                                                     batch_size=batch_size,
                                                     collate_fn=choice_collate_func,
                                                     shuffle=shuffle)
    return choice_data_loader, len(choice_data)


"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/11/10 8:46
"""
import numpy as np
import os
import jieba
import json
import torch
import pickle as pkl
from torch.utils.data import Dataset
from BERT.transformers.transformers.tokenization_bert import BertTokenizer



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

    def set_bert_idxs(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


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
                        output.append(ChoiceDatum(raw_text1=content, raw_text2=alter,
                                                  label=label, question_id=idsTemp))
                else:
                    label = 0
                    output.append(ChoiceDatum(raw_text1=content, raw_text2=alter,
                                              label=label, question_id=idsTemp))
    return output


def convert_features(contentA, contentB, tokenizer, max_length_seq, max_length_query):
    # 根据词典，将分好词的句子中token转化成dict_id
    max_length_contentA = max_length_seq - max_length_query - 3
    max_length_contentB = max_length_query
    contentA_token = tokenizer.tokenize(contentA)
    contentB_token = tokenizer.tokenize(contentB)
    if len(contentA_token) > max_length_contentA:
        contentA_token = contentA_token[:int(max_length_contentA)]
    else:
        contentA_token = contentA_token
    if len(contentB_token) > max_length_contentB:
        contentB_token = contentB_token[:int(max_length_contentB)]
    else:
        contentB_token = contentB_token
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in contentA_token:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in contentB_token:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_length_seq:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_length_seq
    assert len(input_mask) == max_length_seq
    assert len(segment_ids) == max_length_seq
    input_ids = np.array(input_ids)
    input_mask = np.array(input_mask)
    segment_ids = np.array(segment_ids)
    return input_ids, input_mask, segment_ids


def process_text_dataset(dataset, tokenizer, max_length_seq, max_length_query):
    # 将token转化成dict_id的数据，加入到选择题数据类中
    for i in range(len(dataset)):
        contentA = dataset[i].raw_text1
        contentB = dataset[i].raw_text2
        input_ids, input_mask, segment_ids = convert_features(contentA, contentB, tokenizer, max_length_seq, max_length_query)
        dataset[i].set_bert_idxs(input_ids, input_mask, segment_ids)
    return dataset


class ChoiceDataset(Dataset):
    # 创建选择题数据集类
    def __init__(self, data_list, max_length_seq, max_length_query):
        # data_list: list of ChoiceDatum
        self.data_list = data_list
        self.max_length_seq = max_length_seq
        self.max_length_query = max_length_query

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        input_ids, input_mask, segment_ids, label = self.data_list[key].input_ids, \
                                                    self.data_list[key].input_mask, \
                                                    self.data_list[key].segment_ids, \
                                                    self.data_list[key].label
        return input_ids, input_mask, segment_ids, label, self.max_length_seq, self.max_length_query


def choice_collate_func(batch):
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    label_list = []
    for datum in batch:
        input_ids_list.append(datum[0])
        input_mask_list.append(datum[1])
        segment_ids_list.append(datum[2])
        label_list.append(datum[3])
    return [torch.LongTensor(np.array(input_ids_list)), torch.LongTensor(np.array(input_mask_list)),
            torch.LongTensor(np.array(segment_ids_list)), torch.LongTensor(np.array(label_list))]


def save(filename, obj, message=None):
    if message is not None:
        print("Save {}...".format(message))
        with open(filename, "wb") as fh:
            pkl.dump(obj, fh)


def prepro(config):
    tokenizer = BertTokenizer.from_pretrained(config.pre_path)
    train_set = construct_dataset(config.train_file)
    dev_set = construct_dataset(config.dev_file)
    train_data = process_text_dataset(train_set, tokenizer, config.max_length_seq, config.max_length_query)
    dev_data = process_text_dataset(dev_set, tokenizer, config.max_length_seq, config.max_length_query)
    save(config.train_eval, train_data, message="train data")
    save(config.dev_eval, dev_data, message="dev data")


def data_loader(path_dir, batch_size, max_length_seq, max_length_query, shuffle=True):
    print("load train_eval ...")
    with open(path_dir, "rb") as fh:
        data = pkl.load(fh)
    # data = eval(data_)
    choice_data = ChoiceDataset(data, max_length_seq, max_length_query)
    # construct data loader
    choice_data_loader = torch.utils.data.DataLoader(dataset=choice_data,
                                                     batch_size=batch_size,
                                                     collate_fn=choice_collate_func,
                                                     shuffle=shuffle)
    return choice_data_loader, len(choice_data)



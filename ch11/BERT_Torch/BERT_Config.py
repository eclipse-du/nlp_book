"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/11/10 12:05
"""

import argparse
import os
from BERT.BERT_Torch.BERT_Prepro import prepro
from BERT.BERT_Torch.BERT_Main import train, predict, test_one

parser = argparse.ArgumentParser()
home = "../../SIAMESE"
pre_path = "../../SIAMESE/data/chinese_roberta_wwm_ext_pytorch/"
train_file = os.path.join(home, "data", "train_data.json")
dev_file = os.path.join(home, "data", "dev_data.json")
test_file = os.path.join(home, "data", "test_data.json")
train_dir = "train"
model_name = "bert"
dir_name = os.path.join(train_dir, model_name)
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(os.path.join(os.getcwd(), dir_name)):
    os.mkdir(os.path.join(os.getcwd(), dir_name))
target_dir = dir_name
save_dir = os.path.join(dir_name, "model")
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
train_eval = os.path.join(target_dir, "train_data.json")
dev_eval = os.path.join(target_dir, "dev_data.json")
save_test_file = os.path.join(target_dir, "dev_result.txt")

parser.add_argument('--train_file', default=train_file, type=str)
parser.add_argument('--dev_file', default=dev_file, type=str)
parser.add_argument('--test_file', default=test_file, type=str)

parser.add_argument('--train_eval', default=train_eval, type=str)
parser.add_argument('--dev_eval', default=dev_eval, type=str)
parser.add_argument('--save_test_file', default=save_test_file, type=str)
parser.add_argument('--save_dir', default=save_dir, type=str)
parser.add_argument('--target_dir', default=target_dir, type=str)
parser.add_argument('--pre_path', default=pre_path, type=str)
parser.add_argument('--log_fname', default="log.log", type=str)

parser.add_argument('--mode', default="train", type=str)

parser.add_argument('--max_length_seq', default=123, type=int)
parser.add_argument('--max_length_query', default=60, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_warmup_steps', default=int(0.1*48000), type=float)
parser.add_argument('--num_train_steps', default=48000, type=float)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--is_gpu', default=True, type=bool)
parser.add_argument('--adam_epsilon', default=1e-8, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--epoch', default=50, type=int)
parser.add_argument('--display_interval', default=300, type=int)
parser.add_argument('--dev_interval', default=1, type=int)
parser.add_argument('--early_stop', default=10, type=int)


def main():
    config = parser.parse_args()
    config.mode = "train"
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "predict":
        predict(config)
    elif config.mode == "test_one":
        test_one(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    main()

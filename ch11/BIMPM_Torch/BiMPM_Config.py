"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/8/31 14:45
"""

import argparse
import os
from BIMPM.BIMPM_Torch.BiMPM_Main import train, predict, test_one
from BIMPM.BIMPM_Torch.BiMPM_Prepro import prepro


parser = argparse.ArgumentParser()
home = "../../SIAMESE"
train_file = os.path.join(home, "data", "train_data.json")
dev_file = os.path.join(home, "data", "dev_data.json")
test_file = os.path.join(home, "data", "test_data.json")
word_vector_file = os.path.join(home, "data", "word_size300.txt")
char_vector_file = os.path.join(home, "data", "char_size300.txt")

print(word_vector_file)
print(char_vector_file)
train_dir = "train"
model_name = "bimpm"
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
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_data.json")
dev_eval = os.path.join(target_dir, "dev_data.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
save_test_file = os.path.join(target_dir, "dev_result.txt")

parser.add_argument('--train_file', default=train_file, type=str)
parser.add_argument('--dev_file', default=dev_file, type=str)
parser.add_argument('--test_file', default=test_file, type=str)
parser.add_argument('--word_vector_file', default=word_vector_file, type=str)
parser.add_argument('--word_dictionary', default=word_dictionary, type=str)
parser.add_argument('--word_emb_file', default=word_emb_file, type=str)
parser.add_argument('--char_vector_file', default=char_vector_file, type=str)
parser.add_argument('--char_dictionary', default=char_dictionary, type=str)
parser.add_argument('--char_emb_file', default=char_emb_file, type=str)

parser.add_argument('--train_eval', default=train_eval, type=str)
parser.add_argument('--dev_eval', default=dev_eval, type=str)
parser.add_argument('--save_test_file', default=save_test_file, type=str)
parser.add_argument('--save_dir', default=save_dir, type=str)
parser.add_argument('--target_dir', default=target_dir, type=str)
parser.add_argument('--log_fname', default="log.log", type=str)

parser.add_argument('--mode', default="train", type=str)
parser.add_argument('--vocab_size', default=114920, type=int)
parser.add_argument('--vocab_char_size', default=20028, type=int)
parser.add_argument('--char_dims', default=300, type=int)
parser.add_argument('--word_dims', default=300, type=int)
parser.add_argument('--char_limit', default=8, type=int)
parser.add_argument('--hidden_dims', default=100, type=int)
parser.add_argument('--char_hidden_size', default=100, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--epsilon', default=1e-6, type=float)
parser.add_argument('--max_length_content', default=60, type=int)
parser.add_argument('--num_perspective', default=20, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--is_gpu', default=True, type=bool)
parser.add_argument('--optimizer', default="Adam", type=str)
parser.add_argument('--epoch', default=50, type=int)
parser.add_argument('--display_interval', default=200, type=int)
parser.add_argument('--dev_interval', default=1, type=int)
parser.add_argument('--early_stop', default=10, type=int)


def main():
    config = parser.parse_args()
    config.mode = "test_one"
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

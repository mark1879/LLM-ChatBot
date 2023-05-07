import torch

# 预定义的token
PAD_token = 0  # 表示padding
SOS_token = 1  # 句子的开始
EOS_token = 2  # 句子的结束

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DATA_DIR = "data"

SENTENCE_MAX_LENGTH = 10
WORD_MIN_FREQUENCY = 3
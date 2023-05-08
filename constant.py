import re
from config import Config


class Constants:
    CORPUS_FILE = "./data/qingyun.tsv"
    SPLIT_RE = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    KEY_UNKNOWN = '</UNK>'
    KEY_SOS = '</SOS>'          # 句子开始
    KEY_EOS = '</EOS>'          # 句子结束
    KEY_PAD = '</PAD>'

from collections import Counter
import json
import jieba
from config import Config
from constant import Constants

pairs = []
with open(Config.corpus_file, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        qa_pairs = []
        values = line.strip("\n").split('\t')
        for value in values:
            sentence = jieba.lcut(Constants.SPLIT_RE.sub("", value))
            sentence = sentence[:Config.max_sentence_len]
            qa_pairs.append(sentence)
            if len(qa_pairs) == 2:
                break

        pairs.append(qa_pairs)

word_freq = Counter()
for pair in pairs:
    word_freq.update(pair[0])
    word_freq.update(pair[1])

words = [w for w in word_freq.keys() if word_freq[w] > Config.min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map[Constants.KEY_UNKNOWN] = len(word_map) + 1
word_map[Constants.KEY_SOS] = len(word_map) + 1
word_map[Constants.KEY_EOS] = len(word_map) + 1
word_map[Constants.KEY_PAD] = 0

print("Total words are: {}".format(len(word_map)))

with open(Config.word_map_file, 'w') as j:
    json.dump(word_map, j)


def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map[Constants.KEY_UNKNOWN]) for word in words] + [word_map[Constants.KEY_PAD]] * (Config.max_sentence_len - len(words))
    return enc_c


def encode_reply(words, word_map):
    enc_c = [word_map[Constants.KEY_SOS]] + [word_map.get(word, word_map[Constants.KEY_UNKNOWN]) for word in words] + [word_map[Constants.KEY_PAD]] + [
        word_map[Constants.KEY_PAD]] * (Config.max_sentence_len - len(words))
    return enc_c


pairs_encoded = []
for pair in pairs:
    qus = encode_question(pair[0], word_map)
    ans = encode_reply(pair[1], word_map)
    pairs_encoded.append([qus, ans])

with open(Config.pairs_encoded_file, 'w') as p:
    json.dump(pairs_encoded, p)

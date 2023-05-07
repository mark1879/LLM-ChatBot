import itertools
import torch
import random
from create_vocabulary import create_voc
from config import EOS_token, PAD_token


# 把句子的词变成ID
def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# indexes_batch是多个长度不同句子(list)，使用zip_longest padding成定长，长度为最长句子的长度。
# 此处转换了 shape，(batch, max_length) -> (max_length， batch)
def zero_padding(indexes_batch, fillvalue=PAD_token):
    return list(itertools.zip_longest(*indexes_batch, fillvalue=fillvalue))


# l是二维的padding后的list
# 返回m和l的大小一样，如果某个位置是padding，那么值为0，否则为1
def binary_matrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# 把输入句子变成ID，然后再padding，同时返回lengths这个list，标识实际长度。
# 返回的 pad_var 是一个LongTensor，shape是(batch, max_length)，
# lengths是一个list，长度为(batch,)，表示每个句子的实际长度。
def input_var(input_batch, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in input_batch]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    print(pad_var.shape)
    return pad_var, lengths


# 对输出句子进行padding，然后用 binary_matrix 得到每个位置是 padding(0) 还是非 padding，
# 同时返回最大最长句子的长度(也就是padding后的长度)
# 返回值 pad_var 是LongTensor，shape是(batch, max_target_length)
# mask 是ByteTensor，shape也是(batch, max_target_length)
def output_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.ByteTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len


# 处理一个batch的pair句对
def batch_to_train_data(voc, pair_batch):
    # 按照句子的长度(词数)排序
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)

    return inp, lengths, output, mask, max_target_len


def load_data():
    voc, pairs = create_voc()

    small_batch_size = 5
    return batch_to_train_data(voc, [random.choice(pairs) for _ in range(small_batch_size)])


if __name__ == "__main__":
    input_variable, lengths, target_variable, mask, max_target_len = load_data()
    print("lengths:", lengths)
    print("max_target_len:", max_target_len)
import torch


class Config:
    corpus_file = "./data/qingyun.tsv"
    word_map_file = "./data/qingyun_wordmap_corpus.json"
    pairs_encoded_file = './data/qingyun_pairs_encoded.json'
    max_sentence_len = 50
    min_word_freq = 5       # 最小词频
    batch_size = 100
    d_model = 512
    heads = 8
    num_layers = 12
    drop_out = 0.1
    shuffle = True
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    epochs = 10

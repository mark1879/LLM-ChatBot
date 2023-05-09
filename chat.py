import torch.utils.data
from utils import *


def evaluate(transformer_, question_, question_mask_, max_len_, word_map_):

    rev_word_map = {v: k for k, v in word_map_.items()}
    transformer_.eval()
    start_token = word_map_[Constants.KEY_SOS]
    encoded = transformer_.encode(question_, question_mask_)
    words = torch.LongTensor([[start_token]]).to(Config.device)

    for step in range(max_len_ - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(Config.device).unsqueeze(0).unsqueeze(0)
        decoded = transformer_.decode(words, target_mask, encoded, question_mask_)
        predictions = transformer_.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim=1)
        next_word = next_word.item()
        if next_word == word_map_[Constants.KEY_EOS]:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(Config.device)], dim=1)  # (1,step+2)

    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()

    sen_idx = [w for w in words if w not in {word_map_[Constants.KEY_SOS]}]
    sentence = ''.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])

    return sentence


if __name__ == "__main__":
    print("\n")
    choice = input("Choose the checkpoint({0}-{1}):".format(0, Config.epochs - 1))
    choice = int(choice)
    if 0 <= choice < Config.epochs:
        ckpt_path = Config.checkpoint_dir + "checkpoint_{0}.pth.tar".format(choice)
        checkpoint = torch.load(ckpt_path)
        transformer = checkpoint['transformer']

        with open(Config.word_map_file, 'r') as j:
            word_map = json.load(j)

        while(True):
            print("\n")
            question = input("用户: ")
            if question == 'quit':
                break

            enc_qus = []
            for word in question:
                enc_qus.append(word_map.get(word,  word_map[Constants.KEY_UNKNOWN]))

            question = torch.LongTensor(enc_qus).to(Config.device).unsqueeze(0)
            question_mask = (question != word_map[Constants.KEY_PAD]).to(Config.device).unsqueeze(1).unsqueeze(1)
            sentence = evaluate(transformer, question, question_mask, Config.max_sentence_len, word_map)

            print("机器人:{0}".format(sentence))

    else:
        print("Illegal choice")



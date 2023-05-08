import json
import torch
from torch.utils.data import Dataset
import torch.utils.data
from models import *
from utils import *
from config import Config
from constant import Constants


def train(train_loader_, transformer_, criterion_, transformer_optimizer_, epoch_):
    transformer_.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader_):

        samples = question.shape[0]

        question = question.to(Config.device)
        reply = reply.to(Config.device)

        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        out = transformer_(question, question_mask, reply_input, reply_input_mask)

        loss = criterion_(out, reply_target, reply_target_mask)

        transformer_optimizer_.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer_.step()

        sum_loss += loss.item() * samples
        count += samples

        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch_, i, len(train_loader_), sum_loss / count))


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
    with open(Config.word_map_file, 'r') as j:
        word_map = json.load(j)

    train_loader = torch.utils.data.DataLoader(MyDataset(),
                                               batch_size=Config.batch_size,
                                               shuffle=Config.shuffle,
                                               pin_memory=True)

    transformer = Transformer(d_model=Config.d_model, heads=Config.heads, num_layers=Config.num_layers,
                              word_map=word_map)
    transformer = transformer.to(Config.device)
    adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    transformer_optimizer = AdamWarmup(model_size=Config.d_model, warmup_steps=4000, optimizer=adam_optimizer)
    criterion = LossWithLS(len(word_map), 0.1)

    for epoch in range(Config.epochs):
        train(train_loader, transformer, criterion, transformer_optimizer, epoch)

        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')

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
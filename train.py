import json
import os

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

    folder = os.path.exists(Config.checkpoint_dir)
    if not folder:
        os.makedirs(Config.checkpoint_dir)

    for epoch in range(Config.epochs):
        train(train_loader, transformer, criterion, transformer_optimizer, epoch)

        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, Config.checkpoint_dir + 'checkpoint_' + str(epoch) + '.pth.tar')

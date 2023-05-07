import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from config import DATA_DIR


#  fields = ["lineID", "characterID", "movieID", "character", "text"]
def load_lines(file_path, fields):
    lines = {}

    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]
            lines[line_obj["lineID"]] = line_obj

    return lines


# fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
# lines = {"lineID": {"lineID":"", "characterID":"", "movieID":"", "character":"", "text":""}
def load_conversations(file_path, lines, fields):
    coversations = []

    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]

            # convert string into list
            line_ids = eval(conv_obj['utteranceIDs'])
            conv_obj["lines"] = []

            for line_id in line_ids:
                conv_obj["lines"].append(lines[line_id])

            coversations.append(conv_obj)

    return coversations


def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conv in conversations:
        # ignore the last sentence, because there is no answer
        for i in range(len(conv["lines"]) - 1):
            input_line = conv["lines"][i]["text"].strip()
            target_line = conv["lines"][i + 1]["text"].strip()
            if len(input_line) > 0 and len(target_line) > 0:
                qa_pairs.append([input_line, target_line])

    return qa_pairs


def generate_formatted_movie_lines():
    print("Processing corpus...")
    lines = load_lines(os.path.join(DATA_DIR, "movie_lines.txt"), ["lineID", "characterID", "movieID", "character", "text"])

    print("Loading conversations...")
    conversations = load_conversations(os.path.join(DATA_DIR, "movie_conversations.txt"), lines, ["character1ID", "character2ID", "movieID", "utteranceIDs"])

    print("writing newly formatted file...")
    delimiter = str(codecs.decode('\t', "unicode_escape"))

    with open(os.path.join(DATA_DIR, "formatted_movie_lines.txt"), 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)


if __name__ == "__main__":
    generate_formatted_movie_lines()
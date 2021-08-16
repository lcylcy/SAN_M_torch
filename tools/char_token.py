import os
import sys
import json


CHAR_UNKNOWN = "<unk>"
CHAR_SOS = "<sos>"
CHAR_EOS = "<eos>"


class CharTokenizer():
    def __init__(self, char_dict_file):
        self.char_to_tokenid_dict = {}   #字符:id
        self.tokenid_to_char_dict = {}   #id:字符

        with open(char_dict_file, encoding='utf-8') as label_file:
            labels = json.load(label_file)
        for i in range(len(labels)):
            self.char_to_tokenid_dict[labels[i]] = i
            self.tokenid_to_char_dict[i] = labels[i]


    def text_to_tokens(self, input_text):
        char_list = input_text.strip().split()
        tokens = []
        unknown_token = self.char_to_tokenid_dict[CHAR_UNKNOWN]
        for char in char_list:
            if char in self.char_to_tokenid_dict:
                tokens.append(self.char_to_tokenid_dict[char])
            else:
                tokens.append(unknown_token)
        return tokens

    def tokens_to_text(self, tokens):
        char_list = [self.tokenid_to_char_dict[tokenid] for tokenid in tokens]
        return "".join(char_list)

    def get_sos_token(self):
        return self.char_to_tokenid_dict[CHAR_SOS]

    def get_eos_token(self):
        return self.char_to_tokenid_dict[CHAR_EOS]

    def get_unk_token(self):
        return self.char_to_tokenid_dict[CHAR_UNKNOWN]

    def get_vocab_size(self):
        return len(self.char_to_tokenid_dict)


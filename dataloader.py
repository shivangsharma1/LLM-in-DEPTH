import torch
import numpy as np
from cleaning_text import DataProcessing
import requests
from torch.utils.data import DataLoader, Dataset


class Data(Dataset):
    def __init__(self, text, context_len = 10, stride = 4):
        self.inputs = []
        self.target = []

        textprocessor = DataProcessing(text)
        self.text = textprocessor.remove_special_symbols(text)
        self.words = textprocessor.remove_punctuation(self.text)
        self.vocab = sorted(set(self.words))

        self.word2index = {word: index for index, word in enumerate(self.vocab)}
        self.index2word = {index: word for index, word in enumerate(self.vocab)}


        for index in range(0, len(self.words) - context_len, stride):
            curinp = self.words[index:index+context_len]
            curtar = self.words[index+1:index+context_len+1]

            self.inputs.append(torch.tensor([self.word2index[word] for word in curinp]))
            self.target.append(torch.tensor([self.word2index[word] for word in curtar]))

    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index:int = 0):
        return self.inputs[index], self.target[index]
    
    def get_vocab_size(self):
        return len(self.vocab)



if __name__ == '__main__':
    text = requests.get('https://www.gutenberg.org/files/35/35-0.txt').text
    data = Data(text)

    dataloader = DataLoader(data, batch_size = 12, shuffle = True, drop_last = True)
    X, y = next(iter(dataloader))

    print("inputs: ", X)
    print()
    print("targets: ", y)
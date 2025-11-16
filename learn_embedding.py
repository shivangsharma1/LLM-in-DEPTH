import requests
import numpy as np
from cleaning_text import DataProcessing

class Embedding:
    def __init__(self, text):
        self.inputs = []
        self.target = []
        textprocessor = DataProcessing(text)
        self.text = textprocessor.remove_special_symbols(text)
        self.words = textprocessor.remove_punctuation(self.text)
        self.vocab = sorted(set(self.words))

        self.word2index = {word: index for index, word in enumerate(self.vocab)}
        self.index2word = {index: word for index, word in enumerate(self.vocab)}

        self.create_dataset()
        
    
    def create_dataset(self, context_len:int = 8, stride:int = 2):
        
        for i in range(0, len(self.words) - context_len, stride):
            cur_inp = self.words[i:i+context_len]
            cur_target = self.words[i+1:i+context_len+1]

            self.inputs.append([self.word2index[i] for i in cur_inp])
            self.target.append([self.word2index[i] for i in cur_target])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index = 0):
        return self.inputs[index], self.target[index]



if __name__ == '__main__':
    text = requests.get('https://www.gutenberg.org/files/35/35-0.txt').text
    embed = Embedding(text)
    
    print(embed[10])

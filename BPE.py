import numpy as np
import re
from collections import Counter

class BPE:
    def __init__(self, text, iters = 10):
        if not isinstance(text, str):
            raise Exception("text must be string")
        
        self.text = text.lower()
        self.chars = sorted(set(self.text))
        self.vocab = {char: index for index, char in enumerate(self.chars)}
        self.index2char = {index: char for index, char in enumerate(self.chars)}

        self.data = list(self.text)
        ids = [self.vocab[c] for c in self.data]

        new_ids = self.train_bpe(ids)
        for _ in range(iters-1):
            new_ids = self.train_bpe(new_ids)

    def get_stats(self, ids):
        id_pair_freq = dict()
        for token1, token2 in zip(ids, ids[1:]):
            id_pair_freq[(token1, token2)] = 1 + id_pair_freq.get((token1, token2), 0)
        return id_pair_freq
    

    def update_vocab(self, most_freq_pair):
        self.vocab[most_freq_pair] = max(self.vocab.values())  + 1

    def replace_ids(self, ids):
        new_ids = [] 
        index = 0
        while index < len(ids):
            if index < len(ids) - 1:
                pair = (ids[index], ids[index+1])
                if pair in self.vocab:
                    new_ids.append(self.vocab[pair])
                    index+=2
                else:
                    new_ids.append(ids[index])
                    index+=1

        return new_ids
        

    def train_bpe(self, ids):
        id_pair_freq = self.get_stats(ids)
        
        most_freq_pair_idx = np.argmax(id_pair_freq .values())
        most_freq_pair = list(id_pair_freq.keys())[most_freq_pair_idx]
        
        self.update_vocab(most_freq_pair)

        new_ids = self.replace_ids(ids)
        return new_ids


    def encoder(self, text):
        pass


    def decoder(self, indices):
        pass




if __name__ == '__main__':
    text = 'like liker love lovely hug hugs hugging hearts'
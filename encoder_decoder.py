import re
from collections import Counter


class EncodeDecode:
    def __init__(self, text):
        self.text = ""
        if isinstance(text, str):
            self.text = text.lower()

        if isinstance(text, list):
            self.text = " ".join(text).lower()

        self.text = re.split(r'\s+', self.text)
        self.vocab = sorted(set(self.text))

        self.word2idx = {word:index for index, word in enumerate(self.vocab)}
        self.idx2word = {index:word for index, word in enumerate(self.vocab)}


    def encoder(self, text):
        text = text.lower()
        text = re.split(r'\s+', text)
        encoding = []
        
        for word in text:
            encoding.append(self.word2idx[word])

        return encoding

    def decoder(self, indices):
        decoded = []
        for indice in indices:
            decoded.append(self.idx2word[indice])
        return " ".join(decoded)
        

if __name__ == '__main__':
    text = [ 'All that we are is the result of what we have thought',
        'To be or not to be that is the question',
        'Be yourself everyone else is already taken' ]
    
    sample = 'to everyone already taken'

    obj = EncodeDecode(" ".join(text))
    print(obj.encoder(sample))
    print(obj.decoder(obj.encoder(sample)))





    

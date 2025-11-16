import string
import numpy as np
import requests
import re

class DataProcessing:
    def __init__(self, text):
        self.text  = ""
        if isinstance(text, str):
            self.text = self.remove_special_symbols(text)
        
        if isinstance(text, list):
            self.text = self.remove_special_symbols(" ".join(text))

        
    def remove_special_symbols(self, text):
        strings2replace = ['\r\n\r\nâ\x80\x9c','â\x80\x9c','â\x80\x9d','\r\n','â\x80\x94','â\x80\x99','â\x80\x98', '_',]
        for match_c in strings2replace:
            regexp = re.compile('%s'%match_c)
            text = regexp.sub(' ', text)
        return self.remove_numer_asci_lower(text)
    
    def remove_numer_asci_lower(self, text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\d+', '', text).lower()
        return text
    
    def remove_punctuation(self, text):
        words = re.split(f'[{string.punctuation}\s]+', text)
        words = [item.strip() for item in words if item.strip()]
        words = [word for word in words if len(word) > 1]
        return words
 

        

if __name__ == '__main__':
    book = requests.get('https://www.gutenberg.org/files/35/35-0.txt')
    text = book.text
    obj = DataProcessing(text)
    


    
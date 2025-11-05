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

        print(self.text[:2000])

    def remove_special_symbols(self, text):
        strings2replace = ['\r\n\r\nâ\x80\x9c','â\x80\x9c','â\x80\x9d','\r\n','â\x80\x94','â\x80\x99','â\x80\x98', '_',]
        for match_c in strings2replace:
            regexp = re.compile('%s'%match_c)
            text = regexp.sub(' ', text)

        return text

            



if __name__ == '__main__':
    book = requests.get('https://www.gutenberg.org/files/35/35-0.txt')
    text = book.text
    obj = DataProcessing(text)
    


    
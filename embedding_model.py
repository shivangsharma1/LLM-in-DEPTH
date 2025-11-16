import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from dataloader import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Embedding(nn.Module):
    def __init__(self, vocab_size, embeddim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embeddim)
        self.linear1 = nn.Linear(context_size * embeddim, 128)
        self.linear2 = nn.Linear(128, vocab_size)


    def forward(self, inputs):
        embeds = self.embedding(inputs).view(inputs.shape[0], -1) #(batch_size, contextlen * embed_dim)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)

        log_probs = F.log_softmax(out, dim = -1)
        return log_probs
    
    

if __name__ == '__main__':
    text = requests.get('https://www.gutenberg.org/files/35/35-0.txt').text
    data = Data(text)
    dataloader = DataLoader(data, batch_size = 32, shuffle = True, drop_last = True)
    vocab_size = data.get_vocab_size()

    model = Embedding(vocab_size=vocab_size, embeddim=100 ,context_size=10)
    X,y = next(iter(dataloader))
    modelOut = model(X)

    # print('Input to model:')
    # print(X), print('')

    # print(f'Output from model (size: {list(modelOut.detach().shape)}):')
    # print(modelOut)

    num_epochs = 25
    total_loss = np.zeros(num_epochs)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)

    pretrained_embedding = model.embedding.weight.detach().cpu().clone()

    for epoch in range(num_epochs):
        print(f"In epoch: {epoch} ")
        epoch_loss = 0

        for X, y in dataloader:
            model.zero_grad()
            out = model(X)
            loss = loss_function(out, y[:, -1])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    
    posttrained_embedding = model.embedding.weight.detach().cpu().clone()
import numpy as np
import torch
import transformers
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

text = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
tokens = tokenizer.encode('\n\n'.join(text['text']), return_tensors = 'pt')
num_tokens = torch.numel(tokens)

model.eval()

seq_len = model.config.n_positions
batch_size = 4

index = torch.randint(num_tokens - seq_len, size=(batch_size,))
X  = tokens[0][index[:,None] + torch.arange(seq_len)]

output = model(X, labels = X)

ppl = torch.exp(output.loss).item() # if an array calculate mean before exponentiating
print("ppl", ppl)

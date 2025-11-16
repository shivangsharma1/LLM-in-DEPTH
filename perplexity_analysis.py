import numpy as np
import torch
import transformers
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, GPT2Tokenizer
# import matplotlib_inline.backend_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

model.eval()
situations = [
    [ 1,1,1,9 ],
    [ 1,1,1,2 ],
    [ 1,1,2,2 ],
    [ 3,1,1,2 ],
    [ 9,1,1,1 ] ]

y = 2 # buidling plot fot 3rd index token (the last one)
plt.figure(figsize=(12, 4))
xlabels =[]

for index, situation in enumerate(situations):
    output = torch.tensor([situation], dtype = torch.float32)
    softmax = torch.exp(output)/torch.sum(torch.exp(output))
    log_softmax = torch.log(softmax)
    print("log_softmax", log_softmax[0])
    loss = -log_softmax[0, y]

    ppl = torch.exp(loss)

    # draw the results
    plt.bar(np.array([.7,.9,1.1,1.3]) + index,output[0],width=.2,edgecolor='k')
    plt.text(1.3+index, output[0,-1] + .1,'Targ',font={'size':14},ha='center',va='bottom')

    # print the results
    print(f'Situation "{index}"')
    print(f'  Raw logits: {[f"{o}" for o in output[0]]}')
    print(f'  Softmax: {[f"{o:.4f}" for o in log_softmax[0]]}')
    print(f'  Loss: {loss.item():.4f}')
    print(f'  Perplexity: {ppl.item():.4f}\n')

    # x-axis tick label
    xlabels.append(f'"{index}"\nppl = {ppl.item():.3f}')


plt.gca().set(title='Model outputs (logits) and perplexity',ylabel='Logits',
            xticks=range(1,len(situations)+1),xticklabels=xlabels,ylim=[0,10])
plt.show()

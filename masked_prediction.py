import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn.functional as F

from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

model.eval()

#with out masking
text = 'Cubism is an art movement that sparked innovations in music and architecture'
masked_text = 'Cubism is an [MASK] movement that sparked innovations in music and architecture'
tokens = tokenizer.encode(masked_text, return_tensors = "pt")

#store index of mask token
mask_index = torch.where(tokens == tokenizer.mask_token_id)[1]
print("masked_text: ", mask_index)

# #printing tokens
# for index, token in enumerate(tokens):
#     print(f"token at index : {index}, is token: {token}")

#forward pass
with torch.no_grad():
    output = model(tokens)

print(f"Output : {output}")
print(f"Output shape: {output.logits.shape}")

# logits
logitsMask = output.logits[0, mask_index, :].squeeze()
maxlogitMask = torch.argmax(logitsMask)


# visualize (logit of token, not preceeding!)
plt.figure(figsize=(10,4))
plt.plot(maxlogitMask,logitsMask[maxlogitMask],'go',markersize=10)
plt.plot(logitsMask,'k.',alpha=.3)

plt.gca().set(title=f'Model prediction is "{tokenizer.decode(maxlogitMask)}" (text is "{tokenizer.decode(tokens[0,mask_index])}")',
            xlabel='Token index',ylabel='Model output logit',xlim=[-10,tokenizer.vocab_size+9])

plt.show()
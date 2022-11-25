from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import torch.nn as nn
import torch

class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        configuration = GPT2Config(n_layer=6,n_head=8, n_embd=512)
        self.gpt_model = GPT2LMHeadModel(configuration)
        # self.model     = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='[PAD]')
    
    def forward(self,x, labels):
        return self.gpt_model(inputs_embeds=x, labels=labels['input_ids'])   # NOt sure how to get  caption out of this as the input is not a tokenized sentence its an embedding

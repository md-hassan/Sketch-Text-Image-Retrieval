from transformers import CLIPTokenizer, CLIPModel
import torch.nn as nn
import torch
import pdb

class encoder_clip16(nn.Module):
    def __init__(self, args):
        super(encoder_clip16, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        if not args.precompute_txt:
            pass
        self.precompute_txt = args.precompute_txt
        self.DEVICE = args.cuda_id

    def forward(self, x, type, wordlevel=False):
        if type == 'sketch' or type == 'image':
            return self.clip_model.get_image_features(x.to(self.DEVICE)) 
        elif type == 'text':
            if self.precompute_txt:
                x['input_ids'] = x['input_ids'].squeeze(1).long()
                x['attention_mask'] = x['attention_mask'].squeeze(1).long()
                return self.clip_model.get_text_features(**x) 
            else:
                if not wordlevel:
                    inputs = self.text_tokenizer(x, padding=True, return_tensors="pt").to(self.DEVICE)
                else:
                    inputs = self.text_tokenizer(x, padding=True, return_tensors="pt").to(self.DEVICE)
                    return torch.stack([self.clip_model.get_text_features(**{'input_ids':x.unsqueeze(dim=-1), 'attention_mask':y.unsqueeze(dim=-1)}) for x, y in zip(inputs['input_ids'], inputs['attention_mask'])], dim=0)
                    # for x, y in zip(inputs['input_ids'], inputs['attention_mask']):
                    #     # print(x.unsqueeze(dim=-1).shape, y.unsqueeze(dim=-1).shape)
                    #     print(self.clip_model.get_text_features(**{'input_ids':x.unsqueeze(dim=-1), 'attention_mask':y.unsqueeze(dim=-1)}).shape)
                    # pdb.set_trace()               
                return self.clip_model.get_text_features(**inputs) 

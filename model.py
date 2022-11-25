from ast import arg
import imghdr
import torch.nn as nn
from decoders.text_decoder import *
from encoders.encoders import encoder_clip16
import pdb



class TaskFormer(nn.Module):
    def __init__(self, args):
        super(TaskFormer, self).__init__()
        if(args.encoder == "clip_16"):
            self.encoder = encoder_clip16(args)
        else:
            print("Undefined encoder {}".format(args.encoder))

        if(args.nlp_decoder == "GPT"):
            self.gpt_decoder = GPT2()

        self.classifier_sketch = ClassificationHead(args)
        self.classifier_text = ClassificationHead(args)
        self.classifier_image = ClassificationHead(args)
        self.DEVICE = args.cuda_id
        # self.emb_dim = args.emb_dim
        # self.hidden_size = args.hidden_size
        # self.no_classes = args.no_classes
        
        # ## initialise MLP for classification

        # self.fc1 = nn.Linear(in_features=self.emb_dim,out_features=self.hidden_size)
        # self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.no_classes)
        # self.relu = nn.ReLU(inplace = True)
        # # self.sigmoid = nn.Sigmoid()
    
    def forward(self, sketch, text, image):
        sketch_embs = self.encoder(sketch, type="sketch")
        text_embs = self.encoder(text, type="text")
        image_embs = self.encoder(image, type="image")

        comb_embs = sketch_embs + text_embs

        gpt_embs = self.encoder(text, type="text", wordlevel=True) + sketch_embs.unsqueeze(dim=1)

        # print(gpt_embs.shape)
        ########### GPT 2 ##################
        # 1. Use clip tokenizer to tokenize text
        with torch.no_grad():
            # gpt_gt   = self.gpt_decoder.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            gpt_gt     = self.encoder.text_tokenizer(text, padding=True, return_tensors="pt").to(self.DEVICE)
        gpt_output   = self.gpt_decoder(gpt_embs,gpt_gt)
        # pdb.set_trace()
        ###################################

        sketch_class = self.classifier_sketch(sketch_embs)
        text_class = self.classifier_text(text_embs)
        image_class = self.classifier_image(image_embs)

        return comb_embs,image_embs,sketch_class,text_class,image_class, gpt_output  #returning everything for now cuz dont know on what are we calculating classficiation loss
    def get_embeddings(self,image=None,sketch=None,text=None):
        comb_embs,sketch_embs,text_embs,image_embs = None,None,None,None

        if(sketch != None):
            sketch_embs = self.encoder(sketch, type="sketch")
        if(text != None):
            text_embs = self.encoder(text, type="text")
        if(image != None):
            image_embs = self.encoder(image, type="image")
        if(sketch!=None and text!=None):
            comb_embs = sketch_embs + text_embs
        elif(sketch!=None):
            comb_embs = sketch_embs
        elif(text!=None):
            comb_embs = text_embs

        return comb_embs,sketch_embs,text_embs,image_embs

class ClassificationHead(nn.Module):
    def __init__(self, args):
        super(ClassificationHead, self).__init__()

        self.emb_dim = args.emb_dim
        self.hidden_size = args.hidden_size
        self.no_classes = args.no_classes

        self.fc1 = nn.Linear(in_features=self.emb_dim,out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.no_classes)
        self.relu = nn.ReLU(inplace = True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x) # ASL function does sigmoid
        return x

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.emb_dim = args.emb_dim
        self.hidden_size = args.hidden_size
        self.no_classes = args.no_classes
        
        self.fc1 = nn.Linear(in_features=self.emb_dim,out_features=self.hidden_size)
        self.fc2 = nn.Linear(in_features=self.hidden_size,out_features=self.no_classes)
        self.relu = nn.ReLU(inplace = True)
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x) # ASL function does sigmoid
        return x


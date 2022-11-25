import imghdr
import wandb
import torch
import argparse
import tqdm as tq
import numpy as np
import torch.nn as nn
from torch import embedding, optim
from utils.utils import *
from model import TaskFormer
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPTokenizer,CLIPModel
import sys
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import json
sys.path.append("/home/shape3d/code/Sketch-Text-Image-Retrieval")
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume_path', default='/home/shape3d/code/Sketch-Text-Image-Retrieval/checkpoints/main/checkpoint_epoch49.pth', type=str)
parser.add_argument('--save_dir', default='checkpoints/main/', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--encoder', default='clip_16', type=str)
parser.add_argument('--nlp_decoder', default='GPT', type=str)
parser.add_argument('--precompute_txt', default=False, type=bool)

# Training setting
parser.add_argument('--freeze_encoder',default=True,type=bool)
parser.add_argument('--freeze_nlp_decoder',default=True,type=bool)
parser.add_argument('--freeze_class_MLP',default=False,type=bool)
parser.add_argument('--MLP_weights',default=None,type=str)
parser.add_argument('--embed_ratio',default=100,type=int, help="weight ratio for contrastive learning task")
parser.add_argument('--cls_ratio',default=10,type=int, help="weight ratio for classification task")
parser.add_argument('--gpt_ratio',default=1,type=int, help="weight ratio for (captioning)")

# CLIP
parser.add_argument('--emb_dim', default=512, type=int,help="output dimension of text and image features")
parser.add_argument("--lr", type=float, default=5.0e-4, help="Learning rate.")
parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
parser.add_argument("--eps", type=float, default=1.0e-6, help="Adam epsilon.")
parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")

# classfication MLP, (RANDOM VALUES IN DEFAULT TO GET MODEL RUNNING FOR NOW)
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--no_classes', default=80, type=int)
parser.add_argument('--cls_weights_path', default='/home/shape3d/code/Sketch-Text-Image-Retrieval/checkpoints/classifiers/_checkpoint_epoch49.pth', type=str)

# GPU
parser.add_argument('--cuda_id', default=1, type=int)

args = parser.parse_args()
IMG_DIR_PATH = '/home/shape3d/code/Sketch-Text-Image-Retrieval/dataset/val2017/images'
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


def store_taskformer_embeds():
    taskformer = TaskFormer(args)
    taskformer = taskformer.to(args.cuda_id)
    checkpoint = torch.load(args.resume_path,map_location="cuda:1")
    start_epoch = checkpoint['epoch']
    taskformer.load_state_dict(checkpoint['model'])
    embeddings = []
    file_paths = {}
    # for file_name in tqdm(os.listdir(IMG_DIR_PATH)):
    for file_name in tqdm(natsorted(glob(IMG_DIR_PATH+'/*'))):
        img_path = file_name
        file_paths[img_path] = []
        img = Image.open(img_path)
        img = processor(images=img, padding=True, return_tensors="pt")
        img = img['pixel_values'][0]
        img = torch.unsqueeze(img,0)
        img = img.to(args.cuda_id)
        _,_,_,image_embs = taskformer.get_embeddings(image = img)
        embeddings.append(image_embs.detach().cpu().numpy())
    with open("/home/shape3d/code/Sketch-Text-Image-Retrieval/embedding_VAL_5K_49.json", "w") as outfile:
        json.dump(file_paths, outfile)
    np.save('/home/shape3d/code/Sketch-Text-Image-Retrieval/embedding_VAL_5K_49.npy',embeddings)

def store_CLIP_embeds():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = clip_model.to("cuda:1")
    embeddings = []
    file_paths = {}
    # for file_name in tqdm(os.listdir(IMG_DIR_PATH)):
    for file_name in tqdm(natsorted(glob(IMG_DIR_PATH+'/*'))):
        img_path = file_name
        file_paths[img_path] = []
        img = Image.open(img_path)
        img = processor(images=img, padding=True, return_tensors="pt")
        img = img['pixel_values'][0]
        img = torch.unsqueeze(img,0)
        img = img.to(args.cuda_id)
        image_embs = clip_model.get_image_features(img)
        embeddings.append(image_embs.detach().cpu().numpy())
    with open("/home/shape3d/code/Sketch-Text-Image-Retrieval/embedding_CLIP_VAL_5K.json", "w") as outfile:
        json.dump(file_paths, outfile)
    np.save('/home/shape3d/code/Sketch-Text-Image-Retrieval/embedding_CLIP_VAL_5K.npy',embeddings)

# store_CLIP_embeds()
store_taskformer_embeds()
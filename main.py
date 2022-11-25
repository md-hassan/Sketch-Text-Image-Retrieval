import wandb
import torch
import argparse
import tqdm as tq
import numpy as np
import torch.nn as nn
from torch import optim
from utils.utils import *
from model import TaskFormer
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume_path', default='checkpoints/main/checkpoint_latest.pth', type=str)
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
parser.add_argument('--cls_weights_path', default='/home/shape3d/code/sketch_text_2d/checkpoints/classifiers/_checkpoint_epoch49.pth', type=str)

# GPU
parser.add_argument('--cuda_id', default=0, type=int)

args = parser.parse_args()

taskformer = TaskFormer(args)

cls_checkpoint = torch.load(args.cls_weights_path)
taskformer.classifier_sketch.load_state_dict(cls_checkpoint['model_sketch'])
taskformer.classifier_text.load_state_dict(cls_checkpoint['model_text'])
taskformer.classifier_image.load_state_dict(cls_checkpoint['model_image'])
taskformer = taskformer.to('cuda')

Dataset = build_dataset
dataset_train = Dataset('train', args)
dataset_val = Dataset('val', args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.RandomSampler(dataset_val)

data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train,drop_last = True)
data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val,drop_last=True)

# from https://github.com/mlfoundations/open_clip (train.py)
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
include = lambda n, p: not exclude(n, p)
named_parameters = list(taskformer.named_parameters())
gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

optimizer = optim.AdamW(
    [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.wd},
    ],
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
)

total_steps = len(data_loader_train) * args.epochs
scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

wandb_id = wandb.util.generate_id()
wandb.init(project="Sketch Text 2D Retrieval", entity="cvig", resume="allow", id=wandb_id, tags=["main"])
start_epoch = 0

for epoch in range(start_epoch, args.epochs):

    taskformer.train()
    total_loss, emb_loss, cls_loss, gpt_loss  = train_one_epoch(taskformer, data_loader_train, epoch, optimizer, scheduler, args)

    taskformer.eval()
    val_total_loss, val_emb_loss, val_cls_loss, val_gpt_loss = evaluate(taskformer, data_loader_val, epoch, args)

    wandb.log({'epoch': epoch, 'Train total loss': total_loss, 'Train embed loss': emb_loss, 
            'Train cls loss': cls_loss, 'Train gpt loss': gpt_loss, 'Val total loss': val_total_loss, 
            'Val embed loss': val_emb_loss, 'Val cls loss': val_cls_loss, 'Val gpt loss': val_gpt_loss, 'GPT loss': gpt_loss})   
    print({'epoch': epoch, 'Train total loss': total_loss, 'Train embed loss': emb_loss, 'Train cls loss': cls_loss, 'Train GPT loss': gpt_loss})

    torch.save({
                'epoch': epoch,
                'model': taskformer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'wandb_id': wandb_id
                }, args.save_dir + 'checkpoint_epoch' + str(epoch) + '.pth')
    wandb.save(args.save_dir + 'checkpoint_epoch' + str(epoch) + '.pth')

wandb.finish()

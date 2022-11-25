import wandb
import torch
import argparse
import tqdm as tq
import numpy as np
import torch.nn as nn
from torch import optim
from utils.utils import *
from model import Classifier
from torch.utils.data import DataLoader
from encoders.encoders import encoder_clip16

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume_path', default='checkpoints/classifiers/checkpoint_latest.pth', type=str)
parser.add_argument('--save_dir', default='checkpoints/classifiers/', type=str)
parser.add_argument('--log_dir', default='logs/classifiers/', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--precompute_txt', default=False, type=bool)

# CLIP
parser.add_argument('--emb_dim', default=512, type=int,help="output dimension of text and image features")
parser.add_argument("--lr", type=float, default=5.0e-4, help="Learning rate.")
parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
parser.add_argument("--eps", type=float, default=1.0e-6, help="Adam epsilon.")
parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
parser.add_argument("--warmup", type=int, default=10000, help="Number of steps to warmup for.")

# classfication MLP
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--no_classes', default=80, type=int)

# GPU
parser.add_argument('--cuda_id', default=0, type=int)

args = parser.parse_args()

encoder = encoder_clip16(args).to('cuda')
for param in encoder.clip_model.parameters(): # freeze encoder weights
    param.requires_grad = False

cls_sketch = Classifier(args).to('cuda')
cls_text = Classifier(args).to('cuda')
cls_image = Classifier(args).to('cuda')

Dataset = build_dataset

dataset_train = Dataset('train', args)
dataset_val = Dataset('val', args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.RandomSampler(dataset_val)

data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train)
data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=sampler_val, drop_last=False)

opt_sketch = optim.Adam(cls_sketch.parameters())
opt_text = optim.Adam(cls_text.parameters())
opt_image = optim.Adam(cls_image.parameters())

wandb_id = wandb.util.generate_id()
wandb.init(project="Sketch Text 2D Retrieval", entity="cvig", resume="allow", id=wandb_id, tags=["pretrain"])

for epoch in range(0, args.epochs):

    classifiers = [cls_sketch, cls_text, cls_image]
    optimizers = [opt_sketch, opt_text, opt_image]

    sketch_loss, text_loss, image_loss  = train_one_epoch_cls(encoder, classifiers, data_loader_train, epoch, optimizers, args)

    val_sketch_loss, val_text_loss, val_image_loss = evaluate_cls(encoder, classifiers, data_loader_val, epoch, args)

    wandb.log({'epoch': epoch, 'Train sketch loss': sketch_loss, 'Val sketch loss': val_sketch_loss,
            'Train text loss': text_loss, 'Val text loss': val_text_loss, 'Train image loss': image_loss, 
            'Val image loss': val_image_loss})

    # writer.add_scalars('pretrain', {'Train sketch loss': sketch_loss.item(), 'Val sketch loss': val_sketch_loss.item(),
    #         'Train text loss': text_loss.item(), 'Val text loss': val_text_loss.item(), 'Train image loss': image_loss.item(), 
    #         'Val image loss': val_image_loss.item()}, epoch)

    print({'Train sketch loss': sketch_loss.item(), 'Val sketch loss': val_sketch_loss.item(),
            'Train text loss': text_loss.item(), 'Val text loss': val_text_loss.item(), 'Train image loss': image_loss.item(), 
            'Val image loss': val_image_loss.item()})


    torch.save({
                'epoch': epoch,
                'model_sketch': cls_sketch.state_dict(),
                'model_text': cls_text.state_dict(),
                'model_image': cls_image.state_dict(),
                'opt_sketch': opt_sketch.state_dict(),
                'opt_text': opt_text.state_dict(),
                'opt_image': opt_image.state_dict(),
                'wandb_id': wandb_id
                }, args.save_dir + '_checkpoint_epoch' + str(epoch) + '.pth')
    wandb.save(args.save_dir + '_checkpoint_epoch' + str(epoch) + '.pth')

wandb.finish()

import os
import sys
import time
import torch
import random
import tqdm as tq
import numpy as np
import torch.nn as nn
from PIL import Image
import pycocotools.coco as coco
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPTokenizer

sys.path.append('/home/shape3d/code/Sketch-Text-Image-Retrieval')

class build_dataset(Dataset):
    def __init__(self, split, args):
        super(build_dataset, self).__init__()

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.precompute_txt = args.precompute_txt
        
        self.data_dir = 'dataset/' + split + '2014/'
        if split == 'train' or split == 'val':
            self.annot_path_caption = os.path.join(self.data_dir, 'annotations/captions_' + split + '2014.json')
            self.annot_path_obj = os.path.join(self.data_dir, 'annotations/instances_' + split + '2014.json')

        self.sketch_dir = self.data_dir + 'sketches/'
        self.image_dir = self.data_dir + 'images/'

        print('==> initializing %s caption data.' % split)
        self.coco_cap = coco.COCO(self.annot_path_caption)
        self.images = self.coco_cap.getImgIds()
        self.num_samples = len(self.images)

        print('==> initializing %s obj data.' % split)
        self.coco_obj = coco.COCO(self.annot_path_obj)

        self.cat2cat = dict()
        for cat in self.coco_obj.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

        self.DEVICE = args.cuda_id
        # print(self.cat2cat)

        # if args.precompute_txt:
        #     t1 = time.time()
        #     self.text_embeds = {}
        #     for index in range(len(self.images)):
        #         img_id = self.images[index]
        #         annotations_cap = self.coco_cap.loadAnns(ids=self.coco_cap.getAnnIds(imgIds=[img_id]))
        #         texts = [t['caption'] for t in annotations_cap]
        #         self.text_embeds[img_id] = [self.text_tokenizer(x, padding=True, return_tensors="pt") for x in texts]
        #     print("Time to precompute text embeddings = {}".format(time.time() - t1))
        #     self.max_text_len = 65 # max text length = 62. Zero pad at end

    def __getitem__(self, index):

        img_id = self.images[index]

        image = Image.open(os.path.join(self.image_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name']))
        image = self.processor(images=image, padding=True, return_tensors="pt")
        image = image['pixel_values'][0]

        sketch = Image.open(os.path.join(self.sketch_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name'][:-3] + 'png'))
        sketch = self.processor(images=sketch, padding=True, return_tensors="pt")
        sketch = sketch['pixel_values'][0]

        if self.precompute_txt:
            text_idx = np.random.choice(np.arange(5))
            text = self.text_embeds[img_id][text_idx]
            pad_len = self.max_text_len - len(self.text_embeds[img_id][text_idx]['input_ids'][0])
            pad = torch.zeros((1, pad_len)).to(self.DEVICE)
            text['input_ids'] = torch.cat((text['input_ids'].to(self.DEVICE), pad), 1)
            text['attention_mask'] = torch.cat((text['attention_mask'].to(self.DEVICE), pad), 1)
            text = text.to(self.DEVICE)
        else:
            annotations_cap = self.coco_cap.loadAnns(ids=self.coco_cap.getAnnIds(imgIds=[img_id]))
            texts = [t['caption'] for t in annotations_cap] # choose randomly from 5 captions
                                                            # might need to change later
            random.shuffle(texts)
            text = texts[0]

        # taken from https://github.com/Alibaba-MIIL/ASL (helper_functions.py)
        annotations_obj = self.coco_obj.loadAnns(ids=self.coco_obj.getAnnIds(imgIds=[img_id]))
        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in annotations_obj:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        cls = output.to(self.DEVICE)

        # ----DEBUG: checking inputs----
        # image = Image.open(os.path.join(self.image_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name']))
        # image.save('temp_image.png')
        # sketch = Image.open(os.path.join(self.sketch_dir, self.coco_cap.loadImgs(ids=[img_id])[0]['file_name'][:-3] + 'png'))
        # sketch.save('temp_sketch.png')
        # print(texts)
        # import json 
        # with open(self.annot_path_obj) as f:
        #     t1 = json.load(f)
        # for obj in annotations_obj:
        #     print(t1['categories'][self.cat2cat[obj['category_id']]])

        return {'sketch': sketch, 'text': text, 'image': image, 'cls': cls}

    def __len__(self):

        return self.num_samples

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

# from https://github.com/mlfoundations/open_clip
def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

# taken from https://github.com/Alibaba-MIIL/ASL
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

# taken from https://github.com/Alibaba-MIIL/ASL
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

loss_query = nn.CrossEntropyLoss()
loss_img = nn.CrossEntropyLoss()
loss_cls = AsymmetricLossOptimized()

def train_one_epoch(taskformer, data_loader_train, epoch, optimizer, scheduler, args):
    taskformer.train()
    avg_epoch_loss, avg_emb_loss, avg_cls_loss, avg_gpt_loss = 0, 0, 0, 0
    for batch_idx, batch in tq.tqdm(enumerate(data_loader_train)):
        # for k in batch:
        #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

        sketch = batch['sketch']
        text = batch['text']
        image = batch['image']
        cls = batch['cls']

        optimizer.zero_grad()
        combined_embs, image_embs, sketch_class, text_class, image_class, gpt_output = taskformer(sketch, text, image)
        logits_per_image = image_embs @ combined_embs.T
        logits_per_query = logits_per_image.T

        ground_truth = torch.arange(args.batch_size, dtype = torch.long, device = 'cuda')
        embed_loss = (loss_query(logits_per_query, ground_truth) + loss_img(logits_per_image, ground_truth)) / 2
        
        cls = cls.max(dim=1)[0]
        cls_loss_sketch = loss_cls(sketch_class, cls)
        cls_loss_text = loss_cls(text_class, cls)
        cls_loss_image = loss_cls(image_class, cls)
        cls_loss = (cls_loss_sketch + cls_loss_text + cls_loss_image) / 3

        total_loss = (args.embed_ratio*embed_loss + args.cls_ratio*cls_loss + args.gpt_ratio * gpt_output.loss) / (args.embed_ratio + args.cls_ratio + args.gpt_ratio)
        total_loss.backward()
        avg_epoch_loss += total_loss
        avg_emb_loss += embed_loss
        avg_cls_loss += cls_loss
        avg_gpt_loss += gpt_output.loss

        optimizer.step()
        step = len(data_loader_train) * epoch + batch_idx
        scheduler(step)

    # print({'epoch': epoch, 'Train total loss': total_loss.item(), 'Train embed loss': embed_loss.item(), 'Train cls loss': cls_loss.item(), 'Train GPT loss': gpt_output.loss})
        

    avg_epoch_loss /= len(data_loader_train)
    avg_emb_loss /= len(data_loader_train)
    avg_cls_loss /= len(data_loader_train)
    avg_gpt_loss /= len(data_loader_train)

    print("Epoch {} of {}. Avg loss = {}".format(epoch, args.epochs, avg_epoch_loss))
    return avg_epoch_loss, avg_emb_loss, avg_cls_loss, avg_gpt_loss

def evaluate(taskformer, data_loader_val, epoch, args):
    with torch.no_grad():
        avg_epoch_loss, avg_emb_loss, avg_cls_loss, avg_gpt_loss = 0, 0, 0, 0
        for _, batch in tq.tqdm(enumerate(data_loader_val)):
            # for k in batch:
            #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

            sketch = batch['sketch']
            text = batch['text']
            image = batch['image']
            cls = batch['cls']

            combined_embs, image_embs, sketch_class, text_class, image_class, gpt_output = taskformer(sketch, text, image)
            logits_per_image = image_embs @ combined_embs.T
            logits_per_query = logits_per_image.T

            ground_truth = torch.arange(args.batch_size, dtype = torch.long, device = 'cuda')
            embed_loss = (loss_query(logits_per_query, ground_truth) + loss_img(logits_per_image, ground_truth)) / 2
            
            cls = cls.max(dim=1)[0]
            cls_loss_sketch = loss_cls(sketch_class, cls)
            cls_loss_text = loss_cls(text_class, cls)
            cls_loss_image = loss_cls(image_class, cls)
            cls_loss = (cls_loss_sketch + cls_loss_text + cls_loss_image) / 3

            total_loss = (args.embed_ratio*embed_loss + args.cls_ratio*cls_loss + args.gpt_ratio*gpt_output.loss) / (args.embed_ratio + args.cls_ratio + args.gpt_ratio)
            avg_epoch_loss += total_loss
            avg_emb_loss += embed_loss
            avg_cls_loss += cls_loss
            avg_gpt_loss += gpt_output.loss

        avg_epoch_loss /= len(data_loader_val)
        avg_emb_loss /= len(data_loader_val)
        avg_cls_loss /= len(data_loader_val)
        avg_gpt_loss /= len(data_loader_val)

        print("Validation @ Epoch {}. Avg loss = {}".format(epoch, avg_epoch_loss))

    return avg_epoch_loss, avg_emb_loss, avg_cls_loss, avg_gpt_loss

def train_one_epoch_cls(encoder, classifiers, data_loader_train, epoch, optimizers, args):
    for c in classifiers:
        c.train()
    encoder.eval()
    avg_sketch_loss, avg_image_loss, avg_text_loss = 0, 0, 0
    for batch_idx, batch in tq.tqdm(enumerate(data_loader_train)):
        # for k in batch:
        #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

        sketch = batch['sketch']
        text = batch['text']
        image = batch['image']
        cls = batch['cls']

        for o in optimizers:
            o.zero_grad()
        with torch.no_grad():
            sketch_embs = encoder(sketch, type="sketch")
            text_embs = encoder(text, type="text")
            image_embs = encoder(image, type="image")

        sketch_class = classifiers[0](sketch_embs)
        text_class = classifiers[1](text_embs)
        image_class = classifiers[2](image_embs)

        cls = cls.max(dim=1)[0]
        cls_loss_sketch = loss_cls(sketch_class, cls) / args.batch_size
        cls_loss_text = loss_cls(text_class, cls) / args.batch_size
        cls_loss_image = loss_cls(image_class, cls) / args.batch_size

        cls_loss_sketch.backward()
        cls_loss_text.backward()
        cls_loss_image.backward()

        avg_sketch_loss += cls_loss_sketch
        avg_text_loss += cls_loss_text
        avg_image_loss += cls_loss_image

        for o in optimizers:
            o.step()

    avg_sketch_loss /= len(data_loader_train)
    avg_text_loss /= len(data_loader_train)
    avg_image_loss /= len(data_loader_train)

    print("Epoch {} of {}\n Avg sketch loss = {}\n Avg text loss = {}\n Avg image loss = {}\n".format(epoch, args.epochs, avg_sketch_loss, avg_text_loss, avg_image_loss))

    return avg_sketch_loss, avg_text_loss, avg_image_loss

def evaluate_cls(encoder, classifiers, data_loader_val, epoch, args):
    with torch.no_grad():
        avg_sketch_loss, avg_image_loss, avg_text_loss = 0, 0, 0
        for _, batch in tq.tqdm(enumerate(data_loader_val)):
            # for k in batch:
            #     batch[k] = batch[k].to(device='cuda', non_blocking=True)

            sketch = batch['sketch']
            text = batch['text']
            image = batch['image']
            cls = batch['cls']

            sketch_embs = encoder(sketch, type="sketch")
            text_embs = encoder(text, type="text")
            image_embs = encoder(image, type="image")

            sketch_class = classifiers[0](sketch_embs)
            text_class = classifiers[1](text_embs)
            image_class = classifiers[2](image_embs)
            
            cls = cls.max(dim=1)[0]
            cls_loss_sketch = loss_cls(sketch_class, cls)
            cls_loss_text = loss_cls(text_class, cls)
            cls_loss_image = loss_cls(image_class, cls)
            # cls_loss = (cls_loss_sketch + cls_loss_text + cls_loss_image) / 3

            avg_sketch_loss += cls_loss_sketch / args.batch_size
            avg_text_loss += cls_loss_text / args.batch_size
            avg_image_loss += cls_loss_image / args.batch_size

        avg_sketch_loss /= len(data_loader_val)
        avg_text_loss /= len(data_loader_val)
        avg_image_loss /= len(data_loader_val)

        print("Validation @ Epoch {}\n Avg sketch loss = {}\n Avg text loss = {}\n Avg image loss = {}\n".format(epoch, avg_sketch_loss, avg_text_loss, avg_image_loss))

    return avg_sketch_loss, avg_text_loss, avg_image_loss

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
from transformers import CLIPProcessor, CLIPTokenizer,CLIPModel
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--resume_path', default='/home/shape3d/code/Sketch-Text-Image-Retrieval/checkpoints/main/checkpoint_epoch49.pth', type=str)
parser.add_argument('--save_dir', default='checkpoints/main/', type=str)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=1, type=int)
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

taskformer = TaskFormer(args)

cls_checkpoint = torch.load(args.cls_weights_path)
taskformer = taskformer.to(args.cuda_id)
checkpoint = torch.load(args.resume_path,map_location="cuda:"+str(args.cuda_id))
start_epoch = checkpoint['epoch']
taskformer.load_state_dict(checkpoint['model'])

image_embeddings = np.load('/home/shape3d/code/Sketch-Text-Image-Retrieval/embedding_VAL_5K_49.npy')
image_embeddings = image_embeddings.squeeze()
image_embeddings = torch.tensor(image_embeddings).to("cuda:1")

with open('/home/shape3d/code/Sketch-Text-Image-Retrieval/embedding_VAL_5K_49.json') as json_file:
    file_names = json.load(json_file).keys()

file_names = np.array(list(file_names))



processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# sketch_path = "/home/shape3d/code/Sketch-Text-Image-Retrieval/sketches/COCO_test2014_000000000016.png"
# text = "image of a person"



clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_model = clip_model.to(args.cuda_id)
text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")


def retrieve_CLIP(sketch_path, text):
    sketch = Image.open(sketch_path)
    sketch = processor(images=sketch, padding=True, return_tensors="pt")
    sketch = sketch['pixel_values'][0]
    sketch = torch.unsqueeze(sketch,0)
    sketch = sketch.to(args.cuda_id)
    sketch_embs = clip_model.get_image_features(sketch) 

    text = text_tokenizer(text, padding=True, return_tensors="pt").to(args.cuda_id)
    text_embs = clip_model.get_text_features(**text)

    combined_embs = sketch_embs + text_embs

    logits_per_image_combined = image_embeddings @ combined_embs.T
    logits_per_image_sketch = image_embeddings @ sketch_embs.T
    logits_per_image_text = image_embeddings @ text_embs.T

    idxs_comb = torch.topk(logits_per_image_combined.squeeze(),10).indices # Are these image indices?? or local indices??
    idxs_sketch = torch.topk(logits_per_image_sketch.squeeze(),10).indices
    idxs_text = torch.topk(logits_per_image_text.squeeze(),10).indices


    idxs_comb = list(idxs_comb.detach().cpu().numpy())
    idxs_sketch = list(idxs_sketch.detach().cpu().numpy())
    idxs_text = list(idxs_text.detach().cpu().numpy())

    return file_names[idxs_comb], file_names[idxs_sketch], file_names[idxs_text]


def retrieve(sketch_path, text):
    sketch = Image.open(sketch_path)
    sketch = processor(images=sketch, padding=True, return_tensors="pt")
    sketch = sketch['pixel_values'][0]
    sketch = torch.unsqueeze(sketch,0)
    sketch = sketch.to(args.cuda_id)

    combined_embs, sketch_embs,text_embs,_ = taskformer.get_embeddings(sketch = sketch, text = text, image = None) # returns combined_embs, sketch_embs,text_embs,image_embs
    logits_per_image_combined = image_embeddings @ combined_embs.T
    logits_per_image_sketch = image_embeddings @ sketch_embs.T
    logits_per_image_text = image_embeddings @ text_embs.T

    idxs_comb = torch.topk(logits_per_image_combined.squeeze(),10).indices # Are these image indices?? or local indices??
    idxs_sketch = torch.topk(logits_per_image_sketch.squeeze(),10).indices
    idxs_text = torch.topk(logits_per_image_text.squeeze(),10).indices


    idxs_comb = list(idxs_comb.detach().cpu().numpy())
    idxs_sketch = list(idxs_sketch.detach().cpu().numpy())
    idxs_text = list(idxs_text.detach().cpu().numpy())

    # # DEBUG
    # name = sketch_path.split('/')[-1].split('.')[0]
    # os.mkdir('results/' + name)
    # os.mkdir('results/' + name + '/sketch/')
    # os.mkdir('results/' + name + '/combined/')
    # os.mkdir('results/' + name + '/text/')

    # shutil.copyfile(sketch_path, "/home/shape3d/code/Sketch-Text-Image-Retrieval/results/" + name + "/sketch.jpg")
    # shutil.copyfile(img_path, "/home/shape3d/code/Sketch-Text-Image-Retrieval/results/" + name + "/image.jpg")
    # with open('/home/shape3d/code/Sketch-Text-Image-Retrieval/results/' + name + '/text.txt', 'w') as f:
    #     f.write(text)

    # print("Combined embedding sketch results")
    # for i,retrival_result in enumerate(file_names[idxs_comb]):
    #     shutil.copyfile(retrival_result, "/home/shape3d/code/Sketch-Text-Image-Retrieval/results/" + name + "/combined/"+str(i)+".jpg")
    #     print(i+1,": ",retrival_result)
    # print()

    # print("Sketch embedding sketch results")
    # for i,retrival_result in enumerate(file_names[idxs_sketch]):
    #     shutil.copyfile(retrival_result, "/home/shape3d/code/Sketch-Text-Image-Retrieval/results/" + name + "/sketch/"+str(i)+".jpg")
    #     print(i+1,": ",retrival_result)
    # print()

    # print("Text embedding sketch results")
    # for i,retrival_result in enumerate(file_names[idxs_text]):
    #     shutil.copyfile(retrival_result, "/home/shape3d/code/Sketch-Text-Image-Retrieval/results/" + name + "/text/"+str(i)+".jpg")
    #     print(i+1,": ",retrival_result)

    # dummy = 1
    return file_names[idxs_comb], file_names[idxs_sketch], file_names[idxs_text]

sketch_base = '/home/shape3d/code/Sketch-Text-Image-Retrieval/dataset/val2017/sketches/'
with open("/home/shape3d/code/Sketch-Text-Image-Retrieval/dataset/val2017/annos/captions_val2017.json") as f:
    cap_annos = json.load(f)

with open("/home/shape3d/code/Sketch-Text-Image-Retrieval/dataset/val2017/annos/instances_val2017.json","r") as f:
    class_annos = json.load(f)

file_id_mappings = {}
for entry in class_annos['images']:
    file_id_mappings[entry['file_name']] = entry['id']

id_to_class_mapping = {}

for entry in class_annos['annotations']:
    try:
        if(entry['category_id'] not in id_to_class_mapping[entry['image_id']]):
            id_to_class_mapping[entry['image_id']].append(entry['category_id'])
    except:
            id_to_class_mapping[entry['image_id']] = [entry['category_id']]

id_to_category = {}
for category in class_annos['categories']:
    id_to_category[category['id']] = category['name']


file_catnames = {}
for file_id in id_to_class_mapping.keys():
    cat_names = []
    for cat in id_to_class_mapping[file_id]:
        cat_names.append(id_to_category[cat])

    file_catnames[file_id] = cat_names

# def get_ids(file_names):

def get_classes(file_paths):
    clsss = {}
    for file_name in file_paths:
        file_name = file_name.split("/")[-1]
        id = file_id_mappings[file_name]
        try:
            clss = id_to_class_mapping[id] #'COCO_val2014_000000357948.jpg'
        except:
            continue
        clsss[id] = clss
    return clsss

def check_topK(true_clss,pred_clss,k):

    # ct = 0
    ki = 1
    for img_id,clss in pred_clss.items():
        if(ki>k):
            return False
        if(len(list(set(list(true_clss.values())[0]) & set(clss))) != 0): # Check for atleast one common class
            return True
        ki+=1
    return False

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

match_comb = {1: 0, 5: 0, 10: 0}# /home/shape3d/code/Sketch-Text-Image-Retrieval/dataset/val2014/images/COCO_val2014_000000357948.jpg
match_sketch = {1: 0, 5: 0, 10: 0}
match_text = {1: 0, 5: 0, 10: 0}
num_non = 0
num_imgs = len(cap_annos['images'])
print(len(cap_annos['images']))
shutil.rmtree('results/')
os.mkdir('results/')
for file in tq.tqdm(cap_annos['images']):
    name = file['file_name']
    id = file['id']

    sketch_path = sketch_base + name[:-4] + '.png'
    img_path = '/home/shape3d/code/Sketch-Text-Image-Retrieval/dataset/val2017/images/' + name
    caps = [x['caption'] for x in cap_annos['annotations'] if x['image_id'] == id]
    random.shuffle(caps)
    text = caps[0]

    file_comb, file_sketch, file_text = retrieve(sketch_path, text)

    cls_true = get_classes([name])
    if(cls_true == {}):
        num_non+=1
        continue
    cls_combs = get_classes(file_comb)
    cls_sketch = get_classes(file_sketch)
    cls_text = get_classes(file_text)

    # text_acc = accuracy(cls_text,cls_true,(1,5,10))

    for k in [1, 5, 10]:
        if(check_topK(cls_true,cls_combs,k)):
            match_comb[k]+=1
    for k in [1, 5, 10]:
        if(check_topK(cls_true,cls_sketch,k)):
            match_sketch[k]+=1
    for k in [1, 5, 10]:
        if(check_topK(cls_true,cls_text,k)):
            match_text[k]+=1


# for file in tq.tqdm(cap_annos['images']):
#     name = file['file_name']
#     id = file['id']

#     sketch_path = sketch_base + name[:-4] + '.png'
#     img_path = '/home/shape3d/code/Sketch-Text-Image-Retrieval/dataset/val2017/images/' + name
#     caps = [x['caption'] for x in cap_annos['annotations'] if x['image_id'] == id]
#     random.shuffle(caps)
#     text = caps[0]
    
#     file_comb, file_sketch, file_text = retrieve(sketch_path, text)
#     cls_true = get_classes([name])
#     if(cls_true == {}):
#         num_non+=1
#         continue
#     cls_combs = get_classes(file_comb)
#     cls_sketch = get_classes(file_sketch)
#     cls_text = get_classes(file_text)

#     for k in [1, 5, 10]:
#         if(check_topK(cls_true,cls_combs,k)):
#             match_comb[k]+=1
#     for k in [1, 5, 10]:
#         if(check_topK(cls_true,cls_sketch,k)):
#             match_sketch[k]+=1
#     for k in [1, 5, 10]:
#         if(check_topK(cls_true,cls_text,k)):
#             match_text[k]+=1

print("Metrics for combined embeddings")
for k, v in match_comb.items():
    print("Top {} Accuract: {}".format(k, v/num_imgs * 100))

print()

print("Metrics for sketch embeddings")
for k, v in match_sketch.items():
    print("Top {} Accuract: {}".format(k, v/num_imgs * 100))

print()

print("Metrics for text embeddings")
for k, v in match_text.items():
    print("Top {} Accuract: {}".format(k, v/num_imgs * 100))

print()

print("Number of images without class: ",num_non)


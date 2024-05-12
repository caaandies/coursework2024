from argparse import ArgumentParser
import torch
from torch import tensor
from torchvision import transforms
import clip
from PIL import Image
import numpy as np
import os
from random import choice
from tqdm import tqdm

from style_prediction import ParameterPrediction
from style_transfer import transformation
from utils import *
from net import vgg, decoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = ArgumentParser()

parser.add_argument('--images_dir', type=str, default='datasets/content_images')

parser.add_argument('--prompts_file', type=str, default='datasets/prompt_dataset.txt')

parser.add_argument('--n_epochs_model', type=int, default=200)

parser.add_argument('--n_epochs_emb', type=int, default=75)

parser.add_argument('--lr_model', type=float, default=1e-3)

parser.add_argument('--lr_emb', type=float, default=1e-3)

parser.add_argument('--n_crops', type=int, default=64)

parser.add_argument('--crop_size', type=int, default=128)

parser.add_argument('--lambda_patch', type=float, default=3)

parser.add_argument('--lambda_dir', type=float, default=1)

parser.add_argument('--thresh', type=float, default=0.7)

parser.add_argument('--bottleneck', type=int, default=256)

parser.add_argument('--output_folder', type=str, default='trained')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

vgg.load_state_dict(torch.load('input/vgg_normalised.pth',  map_location=device))
vgg = disable_gradients(vgg)
vgg.eval()
encoder = vgg[:31]

decoder.load_state_dict(torch.load('input/decoder.pth', map_location=device))
decoder.eval()

with open(args.prompts_file, 'r') as file:
    prompts = [line.strip() for line in file.readlines()]

images = os.listdir(args.images_dir)

cropper = transforms.Compose([
    transforms.RandomCrop(args.crop_size)
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])

clip_model, preprocess = clip.load('input/ViT-B-32.pt', device=device)
clip_model = clip_model.float()

source = 'a Photo'
with torch.no_grad():
    tokens_source = clip.tokenize(source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source /= text_source.norm(dim=-1, keepdim=True)
    print('Encoding prompts...')
    prompt_embs = []
    for prompt in tqdm(prompts):
        tokens = clip.tokenize(prompt).to(device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        prompt_embs.append(text_features)

target_means = []
target_stds = []

print('Training embeddings...')
for prompt_emb in tqdm(prompt_embs):
    content_img = load_image(os.path.join(args.images_dir, choice(images))).to(device)
    weight_prediction = ParameterPrediction(args.bottleneck)
    weight_prediction.train()
    weight_prediction.to(device)
    bias_prediction = ParameterPrediction(args.bottleneck)
    bias_prediction.train()
    bias_prediction.to(device)

    all_params = list(bias_prediction.parameters()) + list(weight_prediction.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr_emb)

    with torch.no_grad():
        source_features = clip_model.encode_image(clip_normalize(content_img, device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    for epoch in range(1, args.n_epochs_emb + 1):
        weight = weight_prediction(prompt_emb)
        bias = bias_prediction(prompt_emb)

        result = transformation(encoder, decoder, content_img, weight, bias)
        # Patch loss

        loss_patch=0 
        img_proc = []
        for n in range(args.n_crops):
            result_crop = cropper(result)
            result_crop = augment(result_crop)
            img_proc.append(result_crop)

        img_proc = torch.cat(img_proc,dim=0)
        img_aug = img_proc

        image_features = clip_model.encode_image(clip_normalize(img_aug, device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        img_direction = (image_features-source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (prompt_emb-text_source).repeat(image_features.size(0),1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_temp[loss_temp<args.thresh] = 0
        loss_patch+=loss_temp.mean()

        # Directional loss
        
        glob_features = clip_model.encode_image(clip_normalize(result, device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
        
        glob_direction = (glob_features-source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
        
        loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        # Total loss

        total_loss = args.lambda_patch * loss_patch + args.lambda_dir * loss_glob

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    target_stds.append(torch.abs(weight))
    target_means.append(bias)

flattened_target_stds = torch.flatten(torch.stack(target_stds)).detach().to(device)
flattened_target_means = torch.flatten(torch.stack(target_means)).detach().to(device)
prompts_embs_tensor = torch.stack(prompt_embs).squeeze(1).detach().to(device)

torch.save(flattened_target_stds, f'{args.output_folder}/target_stds.pt')
torch.save(flattened_target_means, f'{args.output_folder}/target_means.pt')
torch.save(prompts_embs_tensor, f'{args.output_folder}/style_embs.pt')


weight_prediction = ParameterPrediction(args.bottleneck)
weight_prediction.train()
weight_prediction.to(device)
bias_prediction = ParameterPrediction(args.bottleneck)
bias_prediction.train()
bias_prediction.to(device)

all_params = list(bias_prediction.parameters()) + list(weight_prediction.parameters())
optimizer = torch.optim.Adam(all_params, lr=args.lr_emb)

print('Training model...')
for epoch in tqdm(range(1, args.n_epochs_model + 1)):
    stds = torch.abs(torch.flatten(weight_prediction(prompts_embs_tensor)))
    means = torch.flatten(bias_prediction(prompts_embs_tensor))

    kl_loss = kl_divergence(flattened_target_means, flattened_target_stds, means, stds)

    optimizer.zero_grad()
    kl_loss.backward()
    optimizer.step()

torch.save(weight_prediction.state_dict(), f'{args.output_folder}/WeightPrediction.pth')
torch.save(bias_prediction.state_dict(), f'{args.output_folder}/BiasPrediction.pth')

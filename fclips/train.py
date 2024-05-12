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

from Ghiasi_style_transfer import Ghiasi
from style_prediction_net import TextStylePredictionNetwork
from utils import *

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

parser.add_argument('--lambda_patch', type=float, default=1500)

parser.add_argument('--lambda_dir', type=float, default=500)

parser.add_argument('--lambda_dis', type=float, default=0.5)

parser.add_argument('--thresh', type=float, default=0.7)

parser.add_argument('--output_folder', type=str, default='trained')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

with open(args.prompts_file, 'r') as file:
    prompts = [line.strip() for line in file.readlines()]

images = os.listdir(args.images_dir)

checkpoint = torch.load('input/checkpoint_Ghiasi.pth', map_location=device)
style_transfer = Ghiasi()
style_transfer.load_state_dict(checkpoint['state_dict_ghiasi'])
style_transfer = disable_gradients(style_transfer)
style_transfer.eval()
style_transfer.to(device)

checkpoint_embeddings = torch.load('input/checkpoint_embeddings.pth')
mean_pbn = checkpoint_embeddings['pbn_embedding_mean'].double().to(device)
covariance_pbn = checkpoint_embeddings['pbn_embedding_covariance'].double().to(device)
dis = Distribution(mean_pbn, covariance_pbn)

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

target = []

print('Training embeddings...')
for prompt_emb in tqdm(prompt_embs):
    content_img = load_image(os.path.join(args.images_dir, choice(images))).to(device)

    style_prediction = TextStylePredictionNetwork()
    style_prediction.train()
    style_prediction.to(device)

    optimizer = torch.optim.Adam(style_prediction.parameters(), lr=args.lr_emb)

    with torch.no_grad():
        source_features = clip_model.encode_image(clip_normalize(content_img, device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    for epoch in range(1, args.n_epochs_emb + 1):
        style_emb = style_prediction(prompt_emb)
        result = style_transfer(content_img, style_emb)

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

        # Distribution loss

        loss_dis = dis.squared_mahalanobis(style_emb)

        # Total loss

        total_loss = args.lambda_patch * loss_patch + args.lambda_dir * loss_glob + args.lambda_dis * loss_dis

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    target.append(style_emb)

target_tensor = torch.stack(target).squeeze(1).detach().to(device)
torch.save(target_tensor, f'{args.output_folder}/target.pt')

prompts_embs_tensor = torch.stack(prompt_embs).squeeze(1).detach().to(device)
torch.save(prompts_embs_tensor, f'{args.output_folder}/style_embs.pt')

style_prediction = TextStylePredictionNetwork()
style_prediction.train()
style_prediction.to(device)

optimizer = torch.optim.Adam(style_prediction.parameters(), lr=args.lr_model)
print('Training model...')
for epoch in tqdm(range(1, args.n_epochs_model + 1)):
    squared_diff = (style_prediction(prompts_embs_tensor) - target_tensor) ** 2
    loss_mse = torch.mean(torch.sum(squared_diff, 1), 0)

    optimizer.zero_grad()
    loss_mse.backward()
    optimizer.step()

torch.save(style_prediction.state_dict(), f'{args.output_folder}/StylePrediction.pth')

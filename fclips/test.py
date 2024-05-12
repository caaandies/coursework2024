from argparse import ArgumentParser
import torch
import clip
from torchvision.transforms import ToPILImage
import numpy as np
import os

from Ghiasi_style_transfer import Ghiasi
from style_prediction_net import TextStylePredictionNetwork
from utils import load_image, Distribution


device = torch.device("cpu")

parser = ArgumentParser()

parser.add_argument('--image_path', type=str, default='datasets/content_images/000000000285.jpg')

parser.add_argument('--text', type=str, default='Purple Knitted')

parser.add_argument('--model_path', type=str, default='trained/StylePrediction.pth')

parser.add_argument('--output_folder', type=str, default='output')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

checkpoint = torch.load('input/checkpoint_Ghiasi.pth', map_location=device)
style_transfer = Ghiasi()
style_transfer.load_state_dict(checkpoint['state_dict_ghiasi'])
style_transfer.eval()
style_transfer.to(device)

style_prediction = TextStylePredictionNetwork()
style_prediction.load_state_dict(torch.load(args.model_path, map_location=device))
style_prediction.eval()
style_prediction.to(device)


clip_model, preprocess = clip.load('input/ViT-B-32.pt', device=device)
clip_model = clip_model.float()

content_img = load_image(args.image_path)
image_name = args.image_path.split('/')[-1]

with torch.no_grad():
    tokens = clip.tokenize(args.text).to(device)
    prompt_emb = clip_model.encode_text(tokens).detach()
    prompt_emb /= prompt_emb.norm(dim=-1, keepdim=True)
    style_emb = style_prediction(prompt_emb)
    img_tensor = style_transfer(content_img, style_emb)
    img_tensor = img_tensor.cpu()
    img_tensor = img_tensor.squeeze(0)
    img_tensor = (img_tensor * 255).byte()
    
    stylized_image = ToPILImage()(img_tensor)
    stylized_image.save(f"{args.output_folder}/{args.text} {image_name}")
    print(f'saved to "{args.output_folder}/{args.text} {image_name}"')

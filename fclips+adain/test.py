from argparse import ArgumentParser
import torch
import clip
from torchvision.transforms import ToPILImage
import numpy as np
import os

from style_prediction import ParameterPrediction
from utils import *
from style_transfer import transformation
from net import vgg, decoder


device = torch.device("cpu")

parser = ArgumentParser()

parser.add_argument('--image_path', type=str, default='datasets/content_images/000000002532.jpg')

parser.add_argument('--text', type=str, default='Purple Knitted')

parser.add_argument('--bias_path', type=str, default='trained/BiasPrediction.pth')

parser.add_argument('--weight_path', type=str, default='trained/WeightPrediction.pth')

parser.add_argument('--bottleneck', type=int, default=256)

parser.add_argument('--output_folder', type=str, default='output')

args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

vgg.load_state_dict(torch.load('input/vgg_normalised.pth',  map_location=device))
vgg = disable_gradients(vgg)
vgg.eval()
encoder = vgg[:31]

decoder.load_state_dict(torch.load('input/decoder.pth', map_location=device))
decoder.eval()

weight_prediction = ParameterPrediction(args.bottleneck)
weight_prediction.load_state_dict(torch.load(args.weight_path, map_location=device))
weight_prediction.eval()
weight_prediction.to(device)
bias_prediction = ParameterPrediction(args.bottleneck)
bias_prediction.load_state_dict(torch.load(args.bias_path, map_location=device))
bias_prediction.eval()
bias_prediction.to(device)


clip_model, preprocess = clip.load('input/ViT-B-32.pt', device=device)
clip_model = clip_model.float()

content_img = load_image(args.image_path)
image_name = args.image_path.split('/')[-1]

with torch.no_grad():
    tokens = clip.tokenize(args.text).to(device)
    prompt_emb = clip_model.encode_text(tokens).detach()
    prompt_emb /= prompt_emb.norm(dim=-1, keepdim=True)
    weight = weight_prediction(prompt_emb)
    bias = bias_prediction(prompt_emb)
    img_tensor = transformation(encoder, decoder, content_img, weight, bias)
    img_tensor = img_tensor.cpu()
    img_tensor = img_tensor.squeeze(0)
    img_tensor = (img_tensor * 255).byte()
    
    stylized_image = ToPILImage()(img_tensor)
    stylized_image.save(f"output/{args.text}.png")
    print(f'saved to "{args.output_folder}/{args.text} {image_name}"')

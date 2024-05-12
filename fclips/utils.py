import torch.nn.functional as F
import torch
from torch import tensor
from PIL import Image
from torchvision import transforms
from torch import linalg as LA


def clip_normalize(image,device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image - mean) / std
    return image


def load_image(img_path: str):
    image = Image.open(img_path)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    image_size = image.size()
    assert image_size[1] == 3, 'incorrect image size'
    return image


def disable_gradients(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class Distribution:
    def __init__(self, mean: tensor, covariance: tensor):
        self.mean = mean
        self.chol_cov = LA.cholesky(covariance)
    
    def squared_mahalanobis(self, x: tensor):
        delta  = x - self.mean
        y = LA.solve(self.chol_cov, delta.T)
        return y.T @ y

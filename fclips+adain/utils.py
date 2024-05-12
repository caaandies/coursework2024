import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


def load_image(img_path: str):    
    image = Image.open(img_path)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    image_size = image.size()
    assert image_size[1] == 3 and image_size[2] % 8 == 0 and image_size[3] % 8 == 0, 'incorrect image size'
    return image


def clip_normalize(image,device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image - mean) / std
    return image


def disable_gradients(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def kl_divergence(mu_true, sigma_true, mu_predicted, sigma_predicted):
    term1 = torch.log(sigma_predicted/sigma_true)
    term2 = (sigma_true**2 + (mu_true - mu_predicted)**2) / (2 * sigma_predicted**2)
    kl_div = term1 + term2 - 0.5
    return torch.mean(kl_div)
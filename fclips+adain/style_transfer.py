import torch

@torch.no_grad()
def calc_mean_std(feat: torch.tensor, eps=1e-10):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    if len(size) == 3:
        return feat.mean(dim=(1, 2)), feat.std(dim=(1, 2)) + eps
    elif len(size) == 4:
        return feat.mean(dim=(2, 3)), feat.std(dim=(2, 3)) + eps
    raise ValueError


@torch.no_grad()
def instance_normalization(feat: torch.tensor):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(-1).unsqueeze(-1)
    normalized_feat = (feat - mean) / std
    return normalized_feat


def transformation(encoder, decoder, content, weight, bias, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = encoder(content)
    norm_f = instance_normalization(content_f)
    new_f = norm_f * weight.unsqueeze(-1).unsqueeze(-1) + bias.unsqueeze(-1).unsqueeze(-1)
    trade_off = new_f * alpha + content_f * (1 - alpha)
    return decoder(trade_off)

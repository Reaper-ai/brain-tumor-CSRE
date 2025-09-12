from io import BytesIO
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from torchvision import transforms
from typing import Dict, Tuple
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

flags = {"t1" : True, "t2" : True}


def preprocess(flair: BytesIO, t1ce: BytesIO) -> Tuple[Tensor, Tensor]:

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])

    if flair:
        flair = Image.open(flair).convert('L')
        t1 = transform(flair)
        flags["t1"] = True
    else:
        t1 = torch.zeros((1, 256, 256))
        flags["t1"] = False

    if t1ce:
        t1ce = Image.open(t1ce).convert('L')
        t2 = transform(t1ce)
        flags["t2"] = True
    else:
        t2 = torch.zeros((1, 256, 256))
        flags["t2"] = False


    return t1, t2



def make_overlay(base_img1 : Tensor, base_img2 : Tensor, mask: Tensor, alpha : int =0.5, cmap: str ="jet"):
    """Overlay mask (H,W) on grayscale base_img (H,W)"""

    if flags["t1"] and flags["t2"]:
        base_img1 = base_img1.squeeze(0).cpu().numpy()
        base_img2 = base_img2.squeeze(0).cpu().numpy()

        base_img = 0.5* (base_img1 + base_img2)
    elif flags["t1"]:
        base_img = base_img1.cpu().numpy()
    elif flags["t2"]:
        base_img = base_img2.squeeze(0).cpu().numpy()


    mask = mask.cpu().numpy()

    base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-8)
    base_rgb = np.stack([base_img]*3, axis=-1)

    cmap_fn = plt.get_cmap(cmap)
    mask_rgba = cmap_fn(mask / (mask.max() + 1e-8))[:, :, :3]  # drop alpha


    overlay = (1 - alpha) * base_rgb + alpha * mask_rgba
    return overlay


def SegmentationPipeline2D(flair : BytesIO, t1ce: BytesIO, model: nn.Module, device) -> Dict[str , Image]:
    d1 , d2 = preprocess(flair, t1ce)
    d1 = d1.to(device)
    d2 = d2.to(device)

    inp = torch.cat((d1, d2), dim=0).unsqueeze(0).to(device)  # [B=1, C=2, H, W]

    model.eval()
    with torch.no_grad():
        pred = model(inp)                        # [B, C, H, W]

    prob = F.softmax(pred, dim=1)            # [B, C, H, W]

    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)  # [B, H, W]
    confidence = prob.max(dim=1).values                    # [B, H, W]
    pred_mask = pred.argmax(dim=1).squeeze()            # [H, W]

    rdict = {
        'seg_overlay': make_overlay(d1, d2, pred_mask, cmap="viridis"),
        'entropy_overlay': make_overlay(d1, d2, entropy.squeeze(0)),
        'confidence_overlay': make_overlay(d1, d2, confidence.squeeze(0), cmap="inferno_r")
    }

    return rdict

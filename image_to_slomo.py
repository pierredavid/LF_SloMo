#Copyright (C) 2020 Pierre David

#!/usr/bin/env python3
import argparse
import os
import os.path
#import ctypes
#from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import model
import dataloader
from flow_interpolation import flow_interp
#import platform

import numpy as np
import math
#from SSIM_PIL import compare_ssim

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help='path of image folder to be converted')
parser.add_argument("--checkpoint", type=str, required=True, help='path of checkpoint for pretrained model')
parser.add_argument("--sf", type=int, default=2, help='Specify scale to be applied to frame rate. Default: 2')
parser.add_argument("--batch_size", type=int, default=1, help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument("--output", type=str, default=".", help='Specify output path. Default: current folder')
args = parser.parse_args()

def psnr(img1, img2, pmax=255):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 10 * math.log10(pmax ** 2 / mse)

def main():
    
    inputPath0 = args.input
    outputPath0 = args.output

    # Initialize transforms
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#    mean = [0.429, 0.431, 0.397]
    mean = [0.5, 0.5, 0.5]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean, std=std)
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    # Temporary fix for issue #7 https://github.com/avinashpaliwal/Super-SloMo/issues/7 -
    # - Removed per channel mean subtraction for CPU.
    if (device.type == "cpu"):
        transform = None
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
    
    # Initialize model   
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    
    train = 2
    videoFrames = dataloader.LFSloMo(root=inputPath0, transform=transform, dim=(1024, 436), randomCropSize=(1024, 436), train=train)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size=args.batch_size, shuffle=False)
    
    flowBackWarp = model.backWarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp = flowBackWarp.to(device)
            
    with torch.no_grad():
        for _, (frame0, _, frame1, F01, F10, _, name) in enumerate(videoFramesloader, 0):
            
            for intermediateIndex in range(1, args.sf):
                time = float(intermediateIndex) / args.sf
                
                I0 = frame0.to(device).float()
                I1 = frame1.to(device).float()
                F_0_1 = F01.to(device).float()
                F_1_0 = F10.to(device).float()
                
                I0_np = np.moveaxis(frame0.numpy(), 0, -1)
                I1_np = np.moveaxis(frame1.numpy(), 0, -1)
                F01_np = np.moveaxis(F01.numpy(), 0, -1)
                F10_np = np.moveaxis(F10.numpy(), 0, -1)
                u,v = flow_interp(F01_np, F10_np, I0_np, I1_np, time)
                FT = np.stack((u,v), axis=-1)
                
                F_t_1 = transforms.ToTensor()((1-time)*FT).to(device).float()
                F_t_0 = transforms.ToTensor()(-time*FT).to(device).float()
    
                # Generate intermediate frames
                g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = flowBackWarp(I1, F_t_1)
                
                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                    
                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                
                V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1   = 1 - V_t_0
                    
                g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)
                
                wCoeff = [1 - time, time]
                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                
                # Save intermediate frame
                for batchIndex in range(args.batch_size):
                    It = TP(Ft_p[batchIndex].cpu().detach()).resize(videoFrames.origDim, Image.BILINEAR)
                    It.save(os.path.join(outputPath0, name[batchIndex] + f"_{intermediateIndex:02}.png"))
#    exit(0)

main()

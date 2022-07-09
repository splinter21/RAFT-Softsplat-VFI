import argparse
import os
import cv2
import glob
import numpy as np
import flowiz as fz
import torch
from PIL import Image
from tqdm import tqdm
from random import sample

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
from core.utils import frame_utils

DEVICE = 'cuda'

def load_image(imfile):
    #读取图片
    img = cv2.imread(imfile)
    img = cv2.resize(img,(480,270))
    #转换为PIL对象
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #转换到numpy
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='./models/raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output', help="flow file")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # index = 0
    # with torch.no_grad():
    #     dir = [os.path.join(args.path,f) for f in os.listdir(args.path) if 'Japan' in f] #大概5100余组
    #     dir = sample(dir,3000) #抽取3000小组
    #     pbar = tqdm(total=len(dir))
    #     for d in dir:
    #         im1 = os.path.join(d,'frame1.jpg')
    #         im2 = os.path.join(d,'frame2.jpg')
    #         im3 = os.path.join(d,'frame3.jpg')

    #         ni0 = np.array(Image.open(im1)).transpose(2, 0, 1).astype(np.uint8)
    #         ni1 = np.array(Image.open(im2)).transpose(2, 0, 1).astype(np.uint8)
    #         ngt = np.array(Image.open(im3)).transpose(2, 0, 1).astype(np.uint8)

    #         i0 = load_image(im1)
    #         i1 = load_image(im2)
    #         i2 = load_image(im3)
    #         padder = InputPadder(i0.shape)
    #         i0,i1,i2 = padder.pad(i0,i1,i2)
    #         _,gti0 = model(i1,i0, iters=20, test_mode=True)
    #         _,gti1 = model(i1,i2, iters=20, test_mode=True)
    #         gti0 = padder.unpad(gti0[0]).permute(1, 2, 0).cpu().numpy()
    #         gti1 = padder.unpad(gti1[0]).permute(1, 2, 0).cpu().numpy()

    #         ft0 = fz.convert_from_flow(gti0,mode='UV').transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    #         ft1 = fz.convert_from_flow(gti1,mode='UV').transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
    #         ft0ft1 = np.vstack((ft0, ft1))
    #         i0i1gt = np.vstack((ni0, ni1, ngt))
    #         np.savez(os.path.join(args.output,'{}.npz'.format(index)), ft0ft1 = ft0ft1, i0i1gt = i0i1gt)
    #         index += 1
    #         pbar.update(1)
    #         # frame_utils.writeFlow(os.path.join(args.output,'gt-i0.flo'), gti0)

    with torch.no_grad():
        im1 = './images/000000444.png'
        im2 = './images/000000445.png'

        i0 = load_image(im1)
        i1 = load_image(im2)
        ni0 = cv2.imread(im1)
        ni1 = cv2.imread(im2)
        ni0 = cv2.resize(ni0,(480,270))
        ni1 = cv2.resize(ni1,(480,270))
        padder = InputPadder(i0.shape)
        i0,i1 = padder.pad(i0,i1)
        _,flow_l = model(i0,i1, iters=20, test_mode=True)
        _,flow_r = model(i1,i0, iters=20, test_mode=True)
        flow_l = padder.unpad(flow_l[0]).permute(1, 2, 0).cpu().numpy()
        flow_r = padder.unpad(flow_r[0]).permute(1, 2, 0).cpu().numpy()
        rgbl = fz.convert_from_flow(flow_l)
        rgbr = fz.convert_from_flow(flow_r)       
        gray1 = cv2.cvtColor(rgbl,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(rgbr,cv2.COLOR_BGR2GRAY)
        # mask1 = np.where(gray1 < 252)
        # mask2 = np.where(gray2 < 252)
        # ni0[mask1] = 0
        # ni1[mask2] = 0
        # ret,left = cv2.threshold(gray1,127,255,cv2.THRESH_BINARY)
        # ret,right = cv2.threshold(gray2,127,255,cv2.THRESH_BINARY)
        ret, left = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        ret, right = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        mask1 = np.where(left < 255)
        mask2 = np.where(right < 255)
        ni0[mask1] = 0
        ni1[mask2] = 0
        cv2.imshow('left',left)
        cv2.imshow('right',right)
        cv2.imshow('ni0',ni0)
        cv2.imshow('ni1',ni1)
        select = ni0 if ni0.mean() < ni1.mean() else ni1
        cv2.imshow('select',select)
        cv2.waitKey()
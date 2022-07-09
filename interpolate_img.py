from SoftSplatModel import SoftSplatBaseline
import torch
import cv2 
import torch

torch.set_grad_enabled(False)

def convert(param):
    return {
        k.replace("module.", "flow_predictor."): v
        for k, v in param.items()
        if "module." in k
    }
softsplat_model = torch.load('./models/SoftSplat_finetunedZ.pth')["model"]
raft_model = convert(torch.load('./models/raft-sintel.pth'))
softsplat_model_keys = list(softsplat_model.keys())
for k in softsplat_model_keys:
    if 'flow_predictor' in k:
        softsplat_model.pop(k)
softsplat_model.update(raft_model)
model = SoftSplatBaseline().cuda()
model.load_state_dict(softsplat_model)

def load_image(imfile):
    img = cv2.imread(imfile)
    img = cv2.resize(img,(864,512))
    H,W,_ = img.shape
    ori_h = H
    ori_w = W
    while H % 32 != 0:
        H += 1
    while W % 32 != 0:
        W += 1
    img = cv2.resize(img,(W,H)) 
    img = torch.from_numpy(img.transpose(2,0,1)).float().cuda()/255.
    return [ori_h,ori_w,img]
    
#光流尺度
scale = 1.0

#加载图片
im1 = './images/01.png'
im2 = './images/02.png'
ori_h,ori_w,i0 = load_image(im1)
_,_,i1 = load_image(im2)

#将图片合并 shape: (B,C,T,H,W)
#Batchsize, Channel, TwoFrames, Height, Width
frame_combine = torch.stack([i0,i1],dim=0).unsqueeze(0).permute(0,2,1,3,4).contiguous()

#需要的时刻t
target_ts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # target t=0-1

#得到输出
framets = model(frame_combine, target_ts, scale)
for i in range(len(framets)):
    framet = framets[i].squeeze(0).permute(1,2,0).cpu().detach().numpy()*255
    cv2.imwrite(f'x{i}.png', cv2.resize(framet,(ori_w,ori_h)))
    cv2.waitKey()
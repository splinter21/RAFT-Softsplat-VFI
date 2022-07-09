import contextlib
import os
from queue import Queue
from SoftSplatModel import SoftSplatBaseline
import torch
import cv2
import _thread
from tqdm import tqdm
import torch
import argparse
import time

parser = argparse.ArgumentParser(description='对图片序列进行补帧')
parser.add_argument('--input', dest='input', type=str, default='input',help='图片目录')
parser.add_argument('--output', dest='output', type=str, default='output',help='保存目录')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='1080P时建议0.5, 小于1080P一律1.0')
parser.add_argument('--times', dest='times', type=int, default=5,help='补x倍帧率')
parser.add_argument('--rbuffer', dest='rbuffer', type=int, default=0,help='读写缓存')
parser.add_argument('--wthreads', dest='wthreads', type=int, default=4,help='写入线程')
args = parser.parse_args()

torch.set_grad_enabled(False)

def convert(param):
    return {
        k.replace("module.", "flow_predictor."): v
        for k, v in param.items()
        if "module." in k
    }

softsplat_model = torch.load('./models/SoftSplat_finetunedZ.pth')["model"]
raft_model = convert(torch.load('./models/raft.pth'))
softsplat_model_keys = list(softsplat_model.keys())
for k in softsplat_model_keys:
    if 'flow_predictor' in k:
        softsplat_model.pop(k)
softsplat_model.update(raft_model)
model = SoftSplatBaseline().cuda()
model.load_state_dict(softsplat_model)

def load_image(imfile):
    img = cv2.imread(imfile)
    H,W,_ = img.shape
    ori_h = H
    ori_w = W
    while H % 32 != 0:
        H += 1
    while W % 32 != 0:
        W += 1
    img = cv2.resize(img,(W,H)) 
    return [ori_h,ori_w,img]

def make_inference(frame_combine,times,scale):
    step = 1 / times
    target_ts = [i * step for i in range(1,times)]
    return model(frame_combine, target_ts, scale)

if not os.path.exists(args.output):
    os.mkdir(args.output)

def clear_write_buffer(user_args, write_buffer):
    while True:
        item = write_buffer.get()
        if item is None:
            break
        num = item[0]
        ori_h = item[1]
        ori_w = item[2]
        content = item[3]
        cv2.imencode('.png', cv2.resize(content,(ori_w,ori_h)))[1].tofile('{}/{:0>9d}.png'.format(user_args.output,num))

def build_read_buffer(dir_path, read_buffer, videogen):
    with contextlib.suppress(Exception):
        for frame in videogen:
            _,_,frame = load_image(os.path.join(dir_path, frame))
            read_buffer.put(frame)
    read_buffer.put(None)

def to_tensor(*imgs):
    return [torch.from_numpy(img.transpose(2,0,1)).float().cuda()/255. for img in imgs]

#加载图片序列
videogen = list(os.listdir(args.input))
tot_frame = len(videogen)
pbar = tqdm(total=tot_frame)
videogen.sort()

h,w,lastframe = load_image(os.path.join(args.input, videogen[0]))
videogen = videogen[1:]

read_buffer = Queue(maxsize=args.rbuffer)
write_buffer = Queue(maxsize=args.rbuffer)
_thread.start_new_thread(build_read_buffer, (args.input, read_buffer, videogen))
for _ in range(args.wthreads):
    _thread.start_new_thread(clear_write_buffer, (args,write_buffer))

cnt = 1
# start inference
write_buffer.put([cnt,h,w,lastframe])

while True:
    frame = read_buffer.get()
    if frame is None:
        break
    #将图片合并 shape: (B,C,T,H,W)
    #Batchsize, Channel, TwoFrames, Height, Width
    frame_combine = torch.stack(to_tensor(lastframe,frame),dim=0).unsqueeze(0).permute(0,2,1,3,4).contiguous()
    output = make_inference(frame_combine,args.times,args.scale)
    lo = 0 if output is None else len(output)
    for x in range(lo):
        output[x] = output[x].squeeze(0).permute(1,2,0).cpu().detach().numpy()*255
        write_buffer.put([cnt,h,w,output[x]])
        cnt += 1
    pbar.update(1)
    lastframe = frame
    write_buffer.put([cnt,h,w,lastframe])
    cnt += 1
write_buffer.put([cnt,h,w,lastframe])

# wait for output
while write_buffer.empty() != True:
    time.sleep(1)
pbar.close()

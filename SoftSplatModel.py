import torch
import torch.nn as nn
from softsplat import Softsplat
from GridNet import GridNet
from torch.nn.functional import interpolate, grid_sample
from einops import repeat
import torch.nn.functional as F
from core.raft import RAFT,RAFT_DEFAULT_ARGS

# convert [0, 1] to [-1, 1]
def preprocess(x):
    return x * 2 - 1


# convert [-1, 1] to [0, 1]
def postprocess(x):
    return torch.clamp((x + 1) / 2, 0, 1)


class BackWarp(nn.Module):
    def __init__(self, clip=True):
        super(BackWarp, self).__init__()
        self.device = torch.device('cuda')
        self.clip = clip

    def forward(self, img, flow):
        b, c, h, w = img.shape
        gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w))
        gridX, gridY = gridX.to(self.device), gridY.to(self.device)

        u = flow[:, 0]  # W
        v = flow[:, 1]  # H

        x = repeat(gridX, 'h w -> b h w', b=b).float() + u
        y = repeat(gridY, 'h w -> b h w', b=b).float() + v

        # normalize
        x = (x / w) * 2 - 1
        y = (y / h) * 2 - 1

        # stacking X and Y
        grid = torch.stack((x, y), dim=-1)

        return grid_sample(img, grid, mode='bilinear', align_corners=True, padding_mode='border') if self.clip else grid_sample(img, grid, mode='bilinear', align_corners=True)


class SoftSplatBaseline(nn.Module):
    def __init__(self, predefined_z=False, act=nn.PReLU):
        super(SoftSplatBaseline, self).__init__()
        self.flow_predictor = torch.nn.DataParallel(RAFT(RAFT_DEFAULT_ARGS())).module
        self.fwarp = Softsplat()
        self.bwarp = BackWarp(clip=False)
        self.feature_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                act(),
                nn.Conv2d(32, 32, 3, 1, 1),
                act()
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 2, 1),
                act(),
                nn.Conv2d(64, 64, 3, 1, 1),
                act()
            ),
            nn.Sequential(
                nn.Conv2d(64, 96, 3, 2, 1),
                act(),
                nn.Conv2d(96, 96, 3, 1, 1),
                act()
            ),
        ])
        self.synth_net = GridNet(dim=32, act=act)
        self.predefined_z = predefined_z
        if predefined_z:
            self.alpha = nn.Parameter(-torch.ones(1))
        else:
            self.v_net = nn.Sequential(
                nn.Conv2d(6, 64, 3, 1, 1),
                act(),
                nn.Conv2d(64, 64, 3, 1, 1),
                act(),
                nn.Conv2d(64, 1, 3, 1, 1)
            )

    def forward(self, x, target_ts, scale):
        x = preprocess(x)
        b = x.shape[0]
        fr0, fr1 = x[:, :, 0], x[:, :, 1]
        fr0_l = F.interpolate(fr0,scale_factor=scale,mode='bilinear',align_corners=False)
        fr1_l = F.interpolate(fr1,scale_factor=scale,mode='bilinear',align_corners=False)
        _, flow_l = self.flow_predictor(fr0_l, fr1_l, iters=20, test_mode=True)
        _, flow_r = self.flow_predictor(fr1_l, fr0_l, iters=20, test_mode=True)
        flow = torch.cat([flow_l,flow_r],dim=0)
        flow = F.interpolate(flow,scale_factor = (1.0/scale)) * (1.0/scale)
        f_lv = torch.cat([fr0, fr1], dim=0)
        pyramid = [f_lv]
        for feat_extractor_lv in self.feature_pyramid:
            f_lv = feat_extractor_lv(f_lv)
            pyramid.append(f_lv)

        # Z importance metric
        brightness_diff = torch.abs(self.bwarp(torch.cat([fr1, fr0], dim=0), flow) - torch.cat([fr0, fr1], dim=0))
        if self.predefined_z:
            z = self.alpha * torch.sum(brightness_diff, dim=1, keepdim=True)
        else:
            z = self.v_net(torch.cat([torch.cat([fr0, fr1]), -brightness_diff], dim=1))

        result = []
        for target_t in target_ts:
            target_t = torch.tensor([target_t]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) #timestep=target_t
            # warping
            n_lv = len(pyramid)
            warped_feat_pyramid = []
            for lv in range(n_lv):
                f_lv = pyramid[lv]
                scale_factor = f_lv.shape[-1] / flow.shape[-1]
                flow_lv = interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=False) * scale_factor
                flow01, flow10 = torch.split(flow_lv, b, dim=0)

                flow0t, flow1t = flow01 * target_t, flow10 * (1 - target_t)

                flowt = torch.cat([flow0t, flow1t], dim=0)
                z_lv = interpolate(z, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                warped_f_lv = self.fwarp(f_lv, flowt, z_lv)
                warped_feat_pyramid.append(warped_f_lv)

            concat_warped_feat_pyramid = []
            for feat_lv in warped_feat_pyramid:
                feat0_lv, feat1_lv = feat_lv.chunk(2, dim=0)
                feat_lv = torch.cat([feat0_lv, feat1_lv], dim=1)
                concat_warped_feat_pyramid.append(feat_lv)
            output = self.synth_net(concat_warped_feat_pyramid)
            result.append(postprocess(output))
        return result

if __name__ == '__main__':
    '''
    Example Usage
    '''
    frame0frame1 = torch.randn([1, 3, 2, 448, 256]).cuda()  # batch size 1, 3 RGB channels, 2 frame input, H x W of 448 x 256
    target_t = torch.tensor([0.5]).cuda()
    model = SoftSplatBaseline().cuda()
    model.load_state_dict(torch.load('./ckpt/SoftSplatBaseline_Vimeo.pth'))

    with torch.no_grad():
        output = model(frame0frame1, target_t)

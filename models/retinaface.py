import torch
import torch.nn as nn
import torch.nn.functional as F

from models.net import EResNet as EResNet
# from models.net import SSH as SSH
from models.net import BiFPN as BiFPN
from models.net import CBAM as CBAM
from models.net import ChannelShuffle2 as ChannelShuffle
from models.net import CPM3_BN as CPM3_BN

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)
    
class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()

        self.cfg = cfg
        self.phase = phase
        backbone = None
        
        if cfg['name'] == 'eresnet':
            backbone = EResNet(width_mult=0.0625)
            self.body = backbone
            in_channels_stage2 = cfg['in_channel']
            in_channels_list = [
                in_channels_stage2 * 2,
                in_channels_stage2 * 2,
                in_channels_stage2 * 2,
            ]
            out_channels = cfg['out_channel']
            
            self.bacbkbone_0_cbam = CBAM(in_channels_list[0], 16)
            self.relu_0 = nn.ReLU()

            self.bacbkbone_1_cbam = CBAM(in_channels_list[1], 16)
            self.relu_1 = nn.ReLU()

            self.bacbkbone_2_cbam = CBAM(in_channels_list[2], 16)
            self.relu_2 = nn.ReLU()

            conv_channel_coef = {
                # the channels of P3/P4/P5.
                0: [in_channels_list[0], in_channels_list[1], in_channels_list[2]],
            }
            self.compound_coef=0
            self.fpn_num_filters = [out_channels]
            self.fpn_cell_repeats = [3]
            repeats = self.fpn_cell_repeats[self.compound_coef]
            max_drop_prob = 0.05 + (self.compound_coef * 0.02)
            dpr = [x.item() for x in torch.linspace(0, max_drop_prob, repeats)]

            
            self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if _ == 0 else False,
                    attention=True if self.compound_coef < 6 else False,
                    drop_prob=dpr[_]
                    )
              for _ in range(self.fpn_cell_repeats[self.compound_coef])])
            
            self.bif_cbam_0 = CBAM(out_channels, 16)
            self.bif_relu_0 = nn.ReLU()

            self.bif_cbam_1 = CBAM(out_channels, 16)
            self.bif_relu_1 = nn.ReLU()

            self.bif_cbam_2 = CBAM(out_channels, 16)
            self.bif_relu_2 = nn.ReLU()

            self.ccpm1 = CPM3_BN(out_channels)
            self.ccpm2 = CPM3_BN(out_channels)
            self.ccpm3 = CPM3_BN(out_channels)
            
            self.ccpm1_cs = nn.Sequential(
                    ChannelShuffle(channels=out_channels,groups=2),
                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels//2, kernel_size = 1, stride = 1, groups = 1, bias = False),
                    nn.BatchNorm2d(out_channels//2),
                    nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, groups=out_channels//2, bias=False),
                    nn.BatchNorm2d(out_channels//2),
                    nn.Conv2d(in_channels = out_channels//2, out_channels = out_channels, kernel_size = 1, stride = 1, groups = 1, bias = False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
            )
            
            self.ccpm2_cs = nn.Sequential(
                    ChannelShuffle(channels=out_channels,groups=2),
                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels//2, kernel_size = 1, stride = 1, groups = 1, bias = False),
                    nn.BatchNorm2d(out_channels//2),
                    nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, groups=out_channels//2, bias=False),
                    nn.BatchNorm2d(out_channels//2),
                    nn.Conv2d(in_channels = out_channels//2, out_channels = out_channels, kernel_size = 1, stride = 1, groups = 1, bias = False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
            )

            self.ccpm3_cs = nn.Sequential(
                    ChannelShuffle(channels=out_channels,groups=2),
                    nn.Conv2d(in_channels = out_channels, out_channels = out_channels//2, kernel_size = 1, stride = 1, groups = 1, bias = False),
                    nn.BatchNorm2d(out_channels//2),
                    nn.Conv2d(in_channels=out_channels//2, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1, groups=out_channels//2, bias=False),
                    nn.BatchNorm2d(out_channels//2),
                    nn.Conv2d(in_channels = out_channels//2, out_channels = out_channels, kernel_size = 1, stride = 1, groups = 1, bias = False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
            )

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        
        if self.cfg['name'] == 'eresnet':
            out = self.body(inputs)
            out = out[1:]
            
            #Backbone_CBAM
            cbam_0 = self.bacbkbone_0_cbam(out[0])
            cbam_1 = self.bacbkbone_1_cbam(out[1])
            cbam_2 = self.bacbkbone_2_cbam(out[2])

            cbam_0 = cbam_0 + out[0]
            cbam_1 = cbam_1 + out[1]
            cbam_2 = cbam_2 + out[2]

            cbam_0 = self.relu_0(cbam_0)
            cbam_1 = self.relu_1(cbam_1)
            cbam_2 = self.relu_2(cbam_2)

            b_cbam = [cbam_0, cbam_1, cbam_2]

            
            #BiFPN
            bifpn = self.bifpn(b_cbam)
            
            #BiFPN_CBAM
            bif_cbam0 = self.bif_cbam_0(bifpn[0])
            bif_cbam1 = self.bif_cbam_1(bifpn[1])
            bif_cbam2 = self.bif_cbam_2(bifpn[2])
            
            bif_cbam0 = bif_cbam0 + bifpn[0]
            bif_cbam1 = bif_cbam1 + bifpn[1]
            bif_cbam2 = bif_cbam2 + bifpn[2]

            bif_c_0 =  self.bif_relu_0(bif_cbam0)
            bif_c_1 =  self.bif_relu_1(bif_cbam1)
            bif_c_2 =  self.bif_relu_2(bif_cbam2)

            bif_cbam = [bif_c_0, bif_c_1, bif_c_2]

            #Context Module
            feature1 = self.ccpm1(bif_cbam[0])
            feature2 = self.ccpm2(bif_cbam[1])
            feature3 = self.ccpm3(bif_cbam[2])
            
            #Channel_Shuffle
            feat1 = self.ccpm1_cs(feature1)
            feat2 = self.ccpm2_cs(feature2)
            feat3 = self.ccpm3_cs(feature3)
           
            features = [feat1, feat2,feat3]
        

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
    

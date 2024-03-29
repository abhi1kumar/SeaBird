import torch.nn as nn
import numpy as np
import torch
from .model_parts import CombinationModule
from . import resnet

class CTRBOX(nn.Module):
    def __init__(self, det_params, det_classes, pretrained, down_ratio, final_kernel, head_conv, pred_y_pix, norm_y_pix):
        super(CTRBOX, self).__init__()
        assert down_ratio in [1, 2, 4, 8, 16]
        self.l1 = 2#int(np.log2(down_ratio))
        base = "resnet18" # "resnet101"
        if base == "resnet18":
            # Resnet 18
            # torch.Size([16, 3, 608, 608])
            # torch.Size([16, 64, 304, 304])
            # torch.Size([16, 64, 152, 152])
            # torch.Size([16, 128, 76, 76])
            # torch.Size([16, 256, 38, 38])
            # torch.Size([16, 512, 19, 19])
            channels = [3, 64, 64, 128, 256, 512]
            self.base_network = resnet.resnet18(pretrained=pretrained)
            # Do NOT downsample anything
            self.base_network.conv1.stride = (1,1)
            self.base_network.maxpool.stride = 1
            self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
            self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
            self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
        else:
            # Resnet 101
            # torch.Size([4, 3, 608, 608])
            # torch.Size([4, 64, 304, 304])
            # torch.Size([4, 256, 152, 152])
            # torch.Size([4, 512, 76, 76])
            # torch.Size([4, 1024, 38, 38])
            # torch.Size([4, 2048, 19, 19])
            channels = [3, 64, 256, 512, 1024, 2048]
            self.base_network = resnet.resnet101(pretrained=pretrained)
            self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
            self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
            self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)

        self.det_params = det_params
        self.det_classes = det_classes
        self.pred_y_pix = pred_y_pix
        self.norm_y_pix = norm_y_pix

        for param in self.det_params:
            dimensions = self.det_params[param]
            if param == 'wh':
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, dimensions, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, dimensions, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
            if 'hm' in param:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(param, fc)


    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base_network(x)
        # for i in x:
        #     print(i.shape)
        # print()
        # import matplotlib.pyplot as plt
        # import os
        # for idx in range(x[1].shape[1]):
        #     temp = x[1][0,idx,:,:]
        #     temp = temp.data.cpu().numpy()
        #     plt.imsave(os.path.join('dilation', '{}.png'.format(idx)), temp)
        c4_combine = self.dec_c4(x[-1], x[-2])
        c3_combine = self.dec_c3(c4_combine, x[-3])
        c2_combine = self.dec_c2(c3_combine, x[-4])

        dec_dict = {}
        for param in self.det_params:
            # print(param, self.__getattr__(param))
            dec_dict[param] = self.__getattr__(param)(c2_combine)
            if 'hm' in param or 'cls' in param:
                dec_dict[param] = torch.sigmoid(dec_dict[param])
            if 'y3d' in param and self.pred_y_pix:
                if self.norm_y_pix:
                    dec_dict[param] = torch.sigmoid(dec_dict[param])
                else:
                    dec_dict[param] = torch.relu(dec_dict[param])
        return dec_dict

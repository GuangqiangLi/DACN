from paper1.TE.DCNN1D.MODEL import DCNN1D
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import math
import pdb
from einops import rearrange, repeat
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        features = F.normalize(features.squeeze(), dim=1).unsqueeze(1)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits 
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability #
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        # print(mean_log_prob_pos)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
def grl_hook(coeff):  
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1
class FE(nn.Module):
    def __init__(self, end_feat, dropout, num_classes):
        super(FE, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )



    def forward(self, x):
        x = self.features(x[:, :, :, :])
        x = x.reshape((x.shape[0], -1, x.shape[3])).transpose(1, 2)
        x, (h_n, c_n) = self.lstm1(x)
        x, (h_n, c_n) = self.lstm2(x)
        feature = h_n.squeeze()
        return feature
class DI(nn.Module):
    def __init__(self, end_feat, dropout, num_classes):
        super(DI, self).__init__()

        self.fc_layer = nn.Linear(200, end_feat)
        self.__in_features = end_feat

    def forward(self, x):
        feature = self.fc_layer(x)
        return feature

    def output_num(self):
        return self.__in_features
class CL(nn.Module):
    def __init__(self, end_feat, dropout, num_classes):
        super(CL, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(end_feat, num_classes)
        )


    def forward(self, x):
        y = self.classifier(x)
        # sim=self.similar(feature)
        return y

    def output_num(self):
        return self.__in_features
class ADAIN(nn.Module):
    def __init__(self, ):
        super(ADAIN, self).__init__()
        self.norm = nn.InstanceNorm1d(1, affine=False)
        #         self.fc_layer=nn.Linear(512, 1024)
        self.fc_layer1 = nn.Linear(512, 512)
        self.fc_layer2 = nn.Linear(512, 512)
        self.norm1 = nn.BatchNorm1d(1)
        self.norm2 = nn.BatchNorm1d(1)

    def forward(self, x, z, y):
        x = x.unsqueeze(1)
        # print(z.squeeze().shape)
        gamma = self.fc_layer1(z.squeeze().cuda())
        beta = self.fc_layer2(y.squeeze().cuda())
        # mean_var.view(mean_var.size(0), 1, 1)
        # gamma=self.norm1(mean_var.unsqueeze(1))
        # gamma=torch.clamp(gamma, min=-0.9,max=0.9)
        #         gamma, beta = torch.chunk(mean_var, chunks=2, dim=0)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        # return (1 + gamma) * self.norm(x).squeeze()
        # return gamma * self.norm(x).squeeze() + beta
        return gamma * self.norm(x).squeeze() + beta

    def output_num(self):
        return self.__in_features
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, dropout):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout1 = nn.Dropout(0.)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(0.)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        # self.max_iter = 10000.0
        self.max_iter = 3300.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)  # 计算系数
            x = x * 1.0
            x.register_hook(grl_hook(coeff))  # 
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
class GFCD(nn.Module):
    def __init__(self, model_path,end_feat=256, dropout=0.5, num_classes=10, adv_hidden_size=300, num_residual_blocks=2):
        super(GFCD, self).__init__()
        self.original_model=torch.load(model_path)
        self.FE = self.original_model.features
        self.DI = nn.Sequential(self.original_model.classifier[0],
                                self.original_model.classifier[1],
                                self.original_model.classifier[2])

        self.CL = self.original_model.classifier[3]
        self.adain = ADAIN()
        self.batchnorm = nn.BatchNorm1d(end_feat * num_classes)
        self.D = AdversarialNetwork(end_feat * num_classes, adv_hidden_size, dropout)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        # self.max_iter = 10000.0
        self.max_iter = 3300.0

    def forward(self, x):
        x_ori_feature = self.FE(x.squeeze())
        x_ori_feature =x_ori_feature.view(x_ori_feature.shape[0],-1)
        if self.training:
            a, b = 0.05, 1.95
            z = (a + (b - a) * torch.rand(x.shape[0], 1)).cuda()
            # z = torch.randn(x.shape[0], 1).cuda()
            y = torch.randn(x.shape[0], 1).cuda()
            x_new_feature1 = self.adain(x_ori_feature, z,y)

            x_ori_feature_bak = x_ori_feature.detach().clone()
            x_new_feature2 = self.adain(x_ori_feature_bak, z,y)
            self.iter_num += 1
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
            x_new_feature2 = x_new_feature2 * 1.0
            x_new_feature2.register_hook(grl_hook(coeff))

            x_class=torch.cat([x_ori_feature, x_new_feature1], 0)
            x_domain=torch.cat([x_ori_feature, x_new_feature2], 0)

            x_di_class = self.DI(x_class)
            x_di_domain = self.DI(x_domain)

            cls_out = self.CL(x_di_class)
            cls_out2 = cls_out.detach().clone()

            op_out = torch.bmm(cls_out2.unsqueeze(2), x_di_domain.unsqueeze(1))
            pred_domain = self.D(self.batchnorm(op_out.view(op_out.size(0), -1)))
            return x_di_class, cls_out, pred_domain
        else:
            x_di = self.DI(x_ori_feature)
            cls_out = self.CL(x_di)
            op_out = torch.bmm(cls_out.unsqueeze(2), x_di.unsqueeze(1))
            pred_domain = self.D(op_out.view(op_out.size(0), -1))
            return x_di, cls_out, pred_domain

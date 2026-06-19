# model_phase1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionBlock(nn.Module):
    """
    Lightweight spatial attention block.
    Input: feature map [B, C, H, W]
    Output: attended features and attention map [B,1,H,W]
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels//2, 1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = self.conv1(x)
        a = self.relu(a)
        a = self.conv2(a)
        a = self.sig(a)  # [B,1,H,W]
        out = x * a
        return out, a

class Phase1ResNet50(nn.Module):
    def __init__(self, emb_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        in_dim = resnet.fc.in_features
        # keep backbone until layer4
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        feat_ch = 2048
        self.att = AttentionBlock(feat_ch)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(feat_ch, emb_dim)

    def forward_backbone(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        feat = self.forward_backbone(x)      # [B, C, H, W]
        feat_att, att_map = self.att(feat)  # apply attention
        pooled = self.pool(feat_att).view(x.size(0), -1)
        emb = self.fc(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb, att_map
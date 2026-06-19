import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Phase2ResNet50(nn.Module):
    def __init__(self, emb_dim=512, pretrained=True):
        super().__init__()

        base = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(2048, emb_dim)

        # 🔹 Abstraction head
        self.abs_head = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 🔴 FIX: grayscale → RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat = self.backbone(x).flatten(1)
        emb = F.normalize(self.fc(feat), dim=1)
        abs_level = self.abs_head(emb)
        return emb, abs_level